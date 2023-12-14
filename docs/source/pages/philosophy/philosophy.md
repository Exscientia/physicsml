---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What is PhysicsML?

## Background

PhysicsML is a package for physics based/related models. It covers the five main pillars of machine learning and is
tailored to models that act on 3d point clouds.

By building on top of [``molflux``](https://exscientia.github.io/molflux/index.html), PhysicsML provides self-contained
access to the machine learning ecosystem to enable you to build machine learning models from scratch.

## The Standard API

One of the main challenges of building machine learning models and keeping up to date with the state-of-the-art is the
variety of APIs and interfaces that different models and model packages follow. Even the same submodules in the same
package can have different APIs. This makes using and comparing the rapidly increasing number of models and
features difficult and time-consuming.

The unifying principle of MolFlux is standardisation. Whether you're trying to extract basic features from data, use a
simple random forest regressor, or trying to train a complicated neural network, the API is the same. The motto is "if
you learn it once, you know it always"! What the standard API also provides is smooth interaction between the different
submodules.

## Modular

Including so much functionality in one package is not trivial and python dependencies can often become daunting. The
PhysicsML package handles this by being highly modular. All you need to do to access more functionality is to install
the relevant dependencies.

The modular nature also makes adding new models, features, and datasets much easier. The robust, but simple, abstractions
can handle models and features from simple to complicated ones.

## Acknowledgements

The ``physicsml`` package has been developed by researchers at Exscientia

* Adam Baskerville
* Aayush Gupta
* Ward Haddadin
* Kavindri Ranasinghe
* Ben Suutari
