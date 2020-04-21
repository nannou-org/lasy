# lasy [![Build Status](https://github.com/nannou-org/lasy/workflows/lasy/badge.svg)](https://github.com/nannou-org/lasy/actions) [![Crates.io](https://img.shields.io/crates/v/lasy.svg)](https://crates.io/crates/lasy) [![Crates.io](https://img.shields.io/crates/l/lasy.svg)](https://github.com/nannou-org/lasy/blob/master/LICENSE-MIT) [![docs.rs](https://docs.rs/lasy/badge.svg)](https://docs.rs/lasy)

A small library dedicated to LASER path optimisation.

The goal for the library is to provide a suite of useful tools for optimising
linear vector images for galvanometer-based LASER projection. To allow *lasy*
LASERs to get more done.

This crate implements the full suite of optimisations covered in *Accurate and
Efficient Drawing Method for Laser Projection* by Purkhet Abderyim et al. These
include Graph optimisation, draw order optimisation, blanking delays and sharp
angle delays. See [the
paper](https://art-science.org/journal/v7n4/v7n4pp155/artsci-v7n4pp155.pdf) for
more details.

Please refer to the [docs](https://docs.rs/lasy) to learn more about how to
apply these optimisations within your code.
