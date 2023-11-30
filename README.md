# TimeStepMusic

This repository presents a (somewhat) novel approach to automatic music generation. Specifically, we use a deep “set autoencoder” pretraining step followed by a deep generative adversarial network (GAN). This document will briefly describe what this means and why various choices have been made, followed by an outline of the code provided and how to use it.


Background

There is a large body of literature on this subject and we will be selective in our overview. The GAN framework, first proposed in Generative Adversarial Networks (Goodfellow et al., 2014), features two neural networks or other deep models, a generator and a discriminator, in which the generator attempts to generate synthetic data that the discriminator cannot distinguish from real data and the discriminator attempts to successfully distinguish the two classes. One can also view a GAN as a generative network with a loss function that is itself learned by a different network. GANs have been used mostly in image synthesis tasks, for which a convolutional architecture is generally appropriate.

Music synthesis models can be divided into those that work with discrete data, such as MIDI files, and those that work with audio (waveform) data directly. As we work with MIDI data, we will largely overlook work done on audio data such as DeepMind’s WaveNet and GANSynth (Engel et al., 2019).

Early implementations of GANs in automatic music synthesis with discrete MIDI input used models such as restricted Boltzmann machines or policy gradient methods (see for example SeqGAN, Yu et al. 2016). Later proposals almost universally embed discrete data into Euclidean space so that now-standard backpropagation methods may be used directly. For example, C-RNN-GAN (Mogren, 2016) used a continuous embedding and an LSTM RNN (long short-term memory recurrent neural network) model in order to sequentially generate MIDI music. It uses a variety of techniques for stabilizing the training of the GAN, including L^2 regularization, a “freezing” scheme to keep either discriminator or generator from getting too strong, and feature mapping, where the generator also tries to match an intermediate layer of the discriminator in order to achieve greater variance and avoid overfitting. The data corpus used was a heterogeneous set of freely available MIDI files.

MidiNet (Yang et al., 2017) used a CNN (convolutional neural network) architecture instead of an RNN, focusing on melody generation. The data set used was “synthetic” in the sense that it was not generated by actual performances, but from composed MIDI files, so that there were well-defined beats and measures that could be fed into the CNN. (A warning: some authors use “synthetic” in this context just to mean a MIDI or similar representation as opposed to waveform data. We will try to avoid using this term to avoid confusion.) Specifically, the MidiNet model treats a measure as a two-dimensional array of data (one dimension is time, and the other is pitch) and uses a 2D convolutional GAN to generate these measures, while another 1D convolutional neural network (the “conditioner”) tries to learn dependencies across measures. MuseGAN (Dong et al., 2017) uses similar techniques to learn multi-track polyphonic music with bass, drums, guitar, piano, and strings.

INCO-GAN (Li et al., 2021) proposes an “inception model” that combines features of RNN and CNN models, and has the ability to generate variable-length music. Transformer-GAN (Neves et al., 2022) proposes a transformer architecture (Vaswani et al., 2017) that trains on music that has been manually labeled with human affective states. All of these models use various training stabilization techniques to improve GAN learning, which can be unstable.


Our goals

In this project, we attempt to generate expressive (i.e., “realistic-sounding”) piano music in the western classical tradition, using the marvelous MAESTRO (MIDI and Audio Edited for Syncrhonous TRacks and Organization) dataset provided by Magenta and deriving from real performances in the International Piano-e-Competition (Hawthorne et al., 2019). Our major constraint is computational resources: all computing has been done by a 2020 MacBook Air with 8GB of RAM and no GPU. As such, we work exclusively with the MIDI data rather than the much larger paired audio data, and there are clear limitations on the type and quantity of training that can be done. It would be extremely interesting to train these models further on more powerful machines, but at the moment this is all I have to work with.

As a subgoal, we want our model to learn expressive pedaling (including standard piano techniques like half damping), because as far as I am aware this has not been done before. We make the decision to use GANs simply because I think they are interesting.


The set autoencoder

To achieve these goals, we propose using a GAN only after an extensive pretraining step using a new architecture called a permutation-invariant set autoencoder (PISA) (Kortvelesy et al., 2023). This autoencoder takes as input a variable-size set of data, but its latent variables are of a fixed size and thus appropriate for input into standard neural network architectures. It uses an attention-like mechanism (key-value pairs) to enforce permutation invariance and a separate mechanism in the decoder to predict the size of the decoded set.

The main reason for using this gadget is that it allows us to treat equal units of time as the basic unit of data for input into the GANs, which would not otherwise be possible for expressive music (since the MIDI file is not divided neatly into bars, tempos fluctuate, etc.). Since a given unit of time can have a variable number of MIDI events taking place in it, if we want to work time step by time step with MIDI events we have to first preprocess the data so that it takes a uniform amount of space. The PISA also imposes permutation invariance, which is a reasonably principled decision in this case because the MIDI events already contain the details of their timing, and ordering them does not add any additional information.

This learned, time-uniform representation makes sense as an input into a convolutional GAN in a way that more naive representations would not. It also seems more principled as an input into RNN architectures as well — for example, an RNN that took a MIDI event as its basic data point has to arbitrarily order simultaneous events and cannot easily impose time invariance. With the PISA, both approaches may be compared.

Although the above seems justification enough, there are at least two other possible reasons why the PISA pretraining step works well for us. One is very simple: given our computational constraints, it is very helpful to first reduce the size of the data we are working with, and the PISA accomplishes that (as would a standard autoencoder). The second is that unsupervised pretraining with an autoencoder can benefit the later GAN training by encoding relevant features in its hidden variables. One can think of the PISA as acting as a regularizer, encouraging the GAN to use the features it has discovered in training. There is no direct evidence that this is actually taking place here, as it is not possible to directly compare to a scenario without the PISA. Some experiments in other contexts have shown the benefits of this approach, but it is not always helpful (see for example the discussion in section 15.1.1 in Deep Learning, Goodfellow et al., 2016). Note that we are doing a fairly naive version of unsupervised pretraining here: rather than “fine-tuning” both the autoencoder and the GAN simultaneously, we are simply training the autoencoder, freezing the parameters, and then training the GAN separately.


The convolutional GAN

The GAN is a fairly standard architecture, with the obvious modification of employing one-dimensional convolutions in place of the two-dimensional convolutions common in image processing tasks. Following the idea of the Wasserstein GAN paper (Arjovsky et al., 2017), we employ a “critic” rather than a “discriminator,” which assigns validity scores to samples rather than assessing directly whether the samples are or real or synthesized and judges the proximity of the generated probability distribution to the empirical distribution by the earth mover’s (EM) distance. Following the suggestion of “Improved Training of Wasserstein GANs” (Gulrajani et al., 2017), we enforce the critic to be Lipshitz via a soft gradient penalty, evaluated on samples that are interpolated between real and synthesized data. The framework as a whole is often referred to as WGAN-GP (Wasserstein GAN with gradient penalty). There are many reasons to prefer this architecture to the standard GAN one, as described in the above papers; I will simply highlight one important reason, which is that a discriminator/critic which is “too powerful” compared to the generator does not saturate and lead to vanishing gradients. In our model, as for many GAN models, the discriminator/critic converges much more quickly than the generator, so being able to continue useful training even after the discriminator/critic becomes very strong is essential.

We do make one small change to the “default” WGAN-GP settings by adjusting the hyper parameter controlling the relative importance of the gradient penalty. The original paper recommends a value of 10, but empirically we find that this is too large and leads to a discriminator which is unable to properly converge even without updating the generator. After some experimentation, we choose a value of 0.1.


Data handling and preprocessing

We use as our dataset all MIDI files from the MAESTRO v3.0.0 database, consisting of a total of 1276 real professional-level performances of music in the western classical tradition (corresponding to about 200 hours of music and about 7 million notes). The MAESTRO dataset comes with a proposed train/validation/test split so that no single composition is placed in multiple subsets, which we use. 

MIDI files consist of a sequence of discrete “events” corresponding to actions that one can take, such as the start or end of a note or pedal press. Each event has a number identifying the note or pedal in question, the time elapsed since the last action (measured in “ticks”, which can be zero), and the intensity of the action. In the case of notes the intensity corresponds to how hard the note is pressed (“velocity”), with zero intensity marking the release of a note. In the case of pedals the intensity corresponds to the amount that the pedal is depressed.

We first (losslessly) transform the data so that an event corresponds to an action with a specified time measured from the start of the piece and a specified duration (and an ID number and an intensity), so that we do not have to consider note or pedal releases as separate events. We then do a “pedal binning” step which groups subsequent events of similar pedal intensities into one event, as otherwise there would be many tiny pedal fluctuations (as intensity is measured from 0 to 127, just slowly releasing a pedal can produce many dozens of events that are aurally indistinguishable.) We then group the events into “chunks” (time steps) of a specific length (default 0.5 seconds) in which the start time of an event is measured from the beginning of the chunk. Unfortunately this process is not completely lossless: due to the structure of the PISA, we must specify in advance a maximum number of events per chunk, and we simply discard any events in excess of this limit. In practice we can select the limit high enough (the selected default is 30) so that very few overflows are discarded but the training is computationally manageable.

Finally, we encode each event as a vector for input into the PISA. Several encoding strategies were implemented and tested, including one-hot-encoding each separate pedal or note ID (of which there are 91) and simply doing nothing (meaning that the input to the PISA would include an ID number in which, for example, 87 represents the highest note on the keyboard and 88 represents the damper pedal). The best results seem to come by encoding whether or not the event is pedal-related as one dimension, with two other dimensions given over to the octave and note value of each note. This softly encourages the representation to consider two notes with the same name but different octaves to be “close” to one another, which is a sort of weak octave invariance. The “continuous” data of start time and event intensity are naively normalized to lie in [-1,1], while the event length data is handled in a somewhat more sophisticated way by fitting to a log-normal distribution and composing with the inverse cdf to normalize.

We do not employ any data augmentation strategies, although there are a couple of obvious things to try (raising or lowering pitches by a few semitones, slowing down or speeding up by small amounts). This is mostly because we achieved adequate results without doing so, and having more data would strain our computational resources. The code is written in such a way that adding in data augmentation would be straightforward.


Training

The PISA is trained with minibatch gradient descent with an ADAM optimizer. The loss function used depends on the selected chunk encoding, but in any case is motivated by pragmatism rather than theoretical well-foundedness (e.g. by applying maximum likelihood estimation to a proposed probabilistic model). In the recommended setting described above, the loss is simply taken to be the sum of the mean squared errors of the note identification, octave identification, pedal identification, and “continuous” data (start time, event length, intensity)

The number of latent dimensions must be specified in advance. Testing suggests that, with the default time step lengths and maximum chunk sizes, about 75 latent dimensions is optimal for quick training and minimal loss. In this scheme, each 0.25 second chunk of music is ultimately encoded as a 75-dimensional vector, and training can be completed in a few dozen epochs. Due to computational resource constraints in training the GAN, we instead opt to use 20 latent dimensions, which does audibly lead to reconstruction error but still leads to interesting training. We use a learning rate of 0.0001 and a batch size of 100, both determined empirically.

The convolutional WGAN-GP is also trained with mini batch gradient descent with an ADAM optimizer. The losses for the discriminator/critic and generator are as specified by the WGAN-GP paper, with two exceptions: as already mentioned, the gradient penalty weight has been substantially reduced to 0.1, and additionally a feature matching term is added to the generator to encourage it to compute some of the same features as the discriminator/critic. In theory, this can be useful because it allows information sharing between the discriminator/critic, which learns more quickly, and the generator, which learns slowly. In practice, I have not conducted enough testing to determine whether or not it is actually helpful in this instance. In order to generate 16 seconds of music, a latent dimension of 50 has been used, but other values may work just as well. A learning rate of 0.0001 seems to perform adequately. As generally recommended for GAN training, a small batch size (10) was used.

GAN training takes a long time, and this is no exception. Interesting results can be achieved after about 10 epochs of training, but more training would almost certainly be better. Each epoch takes about eight hours of training on my terrible CPU, so substantial additional training is not feasible for me at the moment.


How to use

Code is located in the python folder. Create folders called 'examples' and 'models' in the directory in which 'python' is located. Also download the MAESTRO dataset v.3.0.0 at https://magenta.tensorflow.org/datasets/maestro#v300, and unzip to the same directory. In order to train the models, first pick appropriate file names and hyperparameters in the header of the file run.py. No other files need to be modified. Open an interactive python environment such as IDLE or a jupyter notebook in this directory, download any necessary packages, and run the following code:

import run
events, annotation, metadata = run.ReadMIDI()
run.TrainAE(events, annotation, metadata)
run.TrainCGAN(events, annotation, metadata)

Also contained in run.py are two methods (ContinueTrainAE and ContinueTrainCGAN) to load an already-existing model from a file and continue training. TrainCGAN will save sample MIDI files from the model every 50000 batches to the “examples” folder.


What else to do

There are two categories of things that I would like to do: things that do not require more computational power, and things that do. In the first category, I would like to implement a data augmentation scheme as described above. More substantively, one benefit of using the PISA is that the resulting encoding can be used with many different potential GAN architectures. I would like to also implement an attention-based RNN GAN, as it would be very interesting to compare the outputs of the two fully trained models: can one hear the difference between convolutional and recurrent networks? An RNN can also be trained with an “end of track” token in order to generate output of variable length, which is not easily possible with a convolutional network.

In the category of things that can be done with more computing power: more training, more tuning of hyperparameters, and experimenting with continuing to train the PISA jointly with the GAN.
