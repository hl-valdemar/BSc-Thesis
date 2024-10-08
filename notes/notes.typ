//==============================================================================
// ifacconf.typ 2023-11-17 Alexander Von Moll
// Template for IFAC meeting papers
//
// Adapted from ifacconf.tex by Juan a. de la Puente
//==============================================================================

// #import "@preview/abiding-ifacconf:0.1.0": *
// #show: ifacconf-rules
// #show: ifacconf.with(
//   title: "Notes for Bachelor's Thesis, C.S.",
//   authors: (
//     (
//       name: "Valdemar H. Lorenzen",
//       email: "valdemar.lorenzen@gmail.com",
//       affiliation: 0,
//     ),
//   ),
//   )

#set math.equation(numbering: "1.")
#set heading(numbering: "I.1.i ::")

#text(size: 20pt, [Notes for Bachelor's Thesis, C.S. & the ISA])

#outline();

// There are a number of predefined theorem-like environments in
// template.typ:
//
// #theorem[ ... ]        // Theorem
// #proof[ ... ]          // Proof
// #lemma[ ... ]          // Lemma
// #claim[ ... ]          // Claim
// #conjecture[ ... ]     // Conjecture
// #corollary[ ... ]      // Corollary
// #fact[ ... ]           // Fact
// #hypothesis[ ... ]     // Hypothesis
// #proposition[ ... ]    // Proposition
// #criterion[ ... ]      // Criterion

= Taming Transformers for High-Resolution Image Synthesis@esser2021tamingtransformershighresolutionimage

== Key Insights

- Taken together, convolutional and transformer architectures can model the compositional nature of our visual world.
- A powerful first stage, which captures as much context as possible in the learned representation, is critical to enable efficient high-resolution image synthesis with transformers.

== Approach

- Instead of representing an image with pixels, they represent it as a composition of perceptually rich image constituents from a codebook.
   - By learning a good code, one can significantly reduce the description length of compositions, which allows for effeciently modelled global interrelations within images with a transformer architecture.

=== Learning an Effective Codebook of Image Constituents for Use in Transformers

- Constituents of an image needs to be expressed in the form of a _sequence_.
- Instead of building individual pixels, complexity necessitates an approach that uses a discrete codebook of learned representations, such that any image $x in RR^(H times W times 3)$ can be represented by a spatial collection of codebook entries $z_q in RR^(h times w times n_z)$, where $n_z$ is the dimensionality of codes.
  - An e*quivalent representation is a sequence of $h dot w$ indices which specify the respective entries in the learned codebook.
- To effectively learn such a discrete spatial codebook, first one can learn a convolutional model consisting of an encoder $E$ and a decoder $G$, such that taken together, they learn to represent images with codes from a learned, discrete codebook $Z = {z_k}^K_(k=1) subset RR^(n_z)$ (refer to @esser2021tamingtransformershighresolutionimage p. 3 for a diagram).
  - One can approximate a given image $x$ by $hat(x) = G(z_q)$.
  - Obtain $z_q$ using the encoding $hat(z) = E(x) in RR^(h times w times n_z)$ and a subsequent element-wise quantization $bold(q)(dot)$ of each spatial code $hat(z)_(i j) in RR^(n_z)$ onto its closest codebook entry $z_k$: $ z_bold(q) = bold(q)(hat(z)) := ( arg min_(z_k in Z) ||hat(z)_(i j) - z_k|| ) in RR^(h times w times n_z). $
  - The reconstruction $hat(x) approx x$ is then given by $ hat(x) = G(z_bold(q)) = G(bold(q)(E(x))). $ <reconstruction>
  - Backpropagation through the non-differentiable quantization operation in Eq. @reconstruction is achieved by a straight-through gradient estimator, which simply copies the gradients from the decoder to the encoder, such that the model and codebook can be trained end-to-end via the loss function $ cal(L)_("VQ")(E, G, Z) = ||x - hat(x)||^2 &+ ||"sg"[E(x)] - z_bold(q)||^2_2 \
  &+ ||"sg"[z_bold(q)] - E(x)||^2_2. $
  - Here, $cal(L)_"rec" = ||x - hat(x)||^2$ is a reconstruction loss, $"sg"[dot]$ denotes the stop-gradient operation, and $||"sg"[z_bold(q)] - E(x)||^2_2$ is the so-called "commitment loss".

==== Learning a Perceptually Rich Codebook

- Use a VQ-GAN, a discriminator and perceptual loss to keep good perceptual quality at increased compression rate.
- Apply a single attention layer on the lowest resolution to aggregate context from everywhere.

=== Learning the Composition of Images with Transformers

==== Latent Transformers

- We can represent images in terms of the codebook-indices with their encodings with $E$ and $G$ available.
- The quantized encoding of an image $x$ is given by $z_bold(q) = bold(q)(E(x)) in RR^(h times w times n_z)$.
  - This is equivalent to a sequence $s in {0, ..., |Z| - 1}^(h times w)$ of indices from the codebook, which is obtained by replacing each code by its index in the codebook $Z$: $ s_(i j) = k "such that" (z_bold(q))_(i j) = z_k. $
- By mapping indices of a sequence $s$ back to their corresponding codebook entries, $z_bold(q) = (z_s_(i j))$ is readily recovered and decoded to an image $hat(x) = G(z_bold(q))$.
- After choosing some ordering of the indices in $s$, image-generation can be formulated as autoregressive next-index prediction: Given indices $s_(<i)$, the transformer learns to predict the distribution of possible next indices, i.e., $p(s_i|s_(<i))$ to compute the likelihood of the full representation as $p(s) = product_i p(s_i|s_(<i))$.
  - This allows for directly maximizing the log-likehood of the data representation: $ cal(L)_"transformer" = EE_(x tilde p(x)) [ - log p(s) ]. $

==== Conditioned Synthesis

- Often a user demands control over the generation process by providing additional information from which an image shall be synthesized.
  - This information, $c$, could be a single label describing the overall image class or another image itself.
- The task is then to learn the likelihood of the sequence given this information: $ p(s|c) = product_i p(s_i|s_(<i), c). $
- If the conditioning information $c$ has spatial extent, we first learn another VQ-GAN to obtain again an index-based representation $r in {0, ..., |Z_c| - 1}^(h_c times w_c)$ with the newly obtained codebook $Z_c$.
  - Due to the autoregressive structure of the transformer, one can then simply prepend $r$ to $s$ and restrict the computation of the negative log-likelihood to entries $p(s_i|s_(<i), r)$.

==== Generating High-Resolution Images

- The attention mechanism of the transformer architecture puts limits on the sequence length $h dot w$ of its inputs $s$.
- To generate images in the megapixel regime, we have to work patch-wise and crop images to restrict the length of $s$ to a maximally feasible size during training.
- To sample images, then use a transformer in a sliding-window manner (see @esser2021tamingtransformershighresolutionimage, p. 5).

= GFowNet-EM for Learning Compositional Latent Variable Models@hu2023gflownetemlearningcompositionallatent

= Other Notes

== Transformers

- Designed to learn long-range interactions on sequential data.
- Expressive but computationally infeasible for long sequences (such as high-resolution images).
  - Quadratic complexity in the sequence length (as all pairwise interactions are taken into account).
- The (self-)attention mechanism can be described by mapping an intermediate representation with three position-wise linear layers into three representations, query $Q in RR^(N times d_k)$, key $K in RR^(N times d_k)$, and value $V in RR^(N times d_v)$, to compute the output as $ "Attn"(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k))V in RR^(N times d_v). $

== Convolutional Neural Networks (CNN)

- Contain certain (inductive) biases:
  - biases that prioritize local interactions;
  - biases towards spatial invariance through the use of shared weights across all positions.
- These biases make them ineffective if a more holistic understanding of the input is required.

== Vartional Autoencoders (VAE)

- Can be used to learn a representation of some data.

=== Vector Quantized Variational Autoencoder (VQ-VAE)

- An approach to learn discrete representations of images.

== Generative Adversarial Networks (GAN)

=== Vector Quantized Generative Adversarial Networks (VQ-GAN)

- We replace the $L_2$ loss for $cal(L)_"rec"$ by a perceptual loss and introduce an adversarial training procedure with a patch-based discriminator $D$ that aims to differentiate between real and reconstructed images: $ cal(L)({ E, G, Z }, D) = [log D(x) + log(1 - D(hat(x)))]. $
- The complete objective for finding optimal compression model $cal(Q)^* = { E^*, G^*, Z^* }$ then reads $ cal(Q)^* = arg min_(E, G, Z) max_D EE_(x tilde p(x)) &[ cal(L)_"VQ"(E, G, Z) \ &+ lambda cal(L)_"GAN"({E, G, Z}, D) ], $ where we compute the adaptive weight $lambda$ according to $ lambda = (gradient_G_L [cal(L)_"rec"]) / (gradient_G_L [ cal(L)_"GAN" ] + delta) $ where $cal(L)_"rec"$ is the perceptual reconstruction loss, $gradient_G_L [dot]$ denotes the gradient of its input w.r.t. the last layer $L$ of the decoder, and $delta = 10^(-6)$ is used for numerical stability.

// Display bibliography.
#bibliography("refs.bib")
