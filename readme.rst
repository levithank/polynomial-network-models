=====================================================================
Π-nets vs. MLPs: A Comparative Analysis of Deep Polynomial Networks on Low-Dimensional Manifolds
=====================================================================

A PyTorch implementation and comparative study of the image generation
experiments described in the CVPR 2020 paper
**"Π-nets: Deep Polynomial Neural Networks"** [1]_. Two toy datasets
(a circle and a spiral) are used to compare how well each architecture
learns the target distribution.

Repository Structure
====================

* ``image_generation_chainer/`` – Runs the original Chainer implementation
  from [1]_, along with an updated ``.yml`` file listing the (older)
  package versions it requires.
* ``image_generation_pytorch_mlp/`` – PyTorch implementation of the
  baseline MLP architecture.
* ``image_generation_pytorch_pnet/`` – PyTorch implementation of the
  Π-net architecture.

Training
========

From inside the folder of the model you want to train, run:

.. code-block:: bash

   python main.py

Requirements
============

Each folder contains a ``.yml`` file listing all required packages. The
MLP and Π-net folders share the same environment.

To create the environment:

.. code-block:: bash

   conda env create -f <environment>.yml

Experiments were run in a Miniconda environment on WSL2.

References
==========

.. [1] G. Chrysos, S. Moschoglou, G. Bouritsas, Y. Panagakis, J. Deng,
   and S. Zafeiriou, "Π-nets: Deep Polynomial Neural Networks," *CVPR*,
   2020. Code: https://github.com/grigorisg9gr/polynomial_nets