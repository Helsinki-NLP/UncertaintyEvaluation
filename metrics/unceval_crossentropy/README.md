---
title: CrossEntropy
emoji: 🤗 
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.19.1
app_file: app.py
pinned: false
tags:
- evaluate
- metric
description: >-
  The cross-entropy of probability distribution q relative to a
  distribution p measure the average number of bits lost if p
  is coded using a coding scheme optimized for q.
---

# Metric Card for CrossEntropy

The cross-entropy of probability distribution p (reference) relative
to a distribution q (prediction) measure the average number of bits
lost if p is coded using a coding scheme optimized for q.

This version assumes a sampled set of q distributions, which are
averaged before comparison to the reference p.
