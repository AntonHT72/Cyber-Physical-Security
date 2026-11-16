# Project A — Attacks on AC state estimation

Start with `project-A-acse.pdf`. `code-guidelines.pdf` describes the repository
the brief requires.

## Contents

| Path | What it is |
|---|---|
| `project-A-acse.pdf` | assignment brief: questions, simulation requirement, assessment |
| `code-guidelines.pdf` | ten rules for the repository and the code |
| `papers/` | the four assigned papers, named by the citation key used in the brief |
| `data/case57.m` | MATPOWER case for the IEEE 57-bus system |
| `code/supporting-code/` | the Lecture 7 exercise code, as a starting point |

## Which paper serves which question

| Key | Paper | Questions |
|---|---|---|
| P0 | Liang, Zhao, Luo, Weller and Dong (2017), review of false data injection attacks | Q2, Q3, Q5, Q6 |
| A1 | Hug and Giampapa (2012), vulnerability of AC state estimation | Q2, Q3, Q4, Q5 |
| A2 | Mohanadhas, Sarma and Mortensen (2016), state estimation at ERCOT | Q1 |
| A3 | Cooper, Bretas and Meyn (2023), anomaly detection in state estimation | Q3, Q6 |

## What the code gives you

`code/supporting-code/` is the Lecture 7 material: DC state estimation on the
IEEE 14-bus system, in Julia with JuliaGrid (Julia 1.10) and in a Python mirror.
Read its README for the environment and the order of the blocks.

It is a starting point, not a template for this project. The AC estimator, the
57-bus measurement set and the three attack constructions are yours to build.
`data/case57.m` is the network only; MATPOWER's `caseformat` documents the fields
if you parse the file yourself.

## Requirements

Julia 1.10 with `code/supporting-code/Project.toml`, or Python 3 with
`code/supporting-code/python/requirements.txt`. The brief also allows Python with
pandapower. The course Python mirror is numpy and scipy based, so that route means
building the estimator against pandapower's own interface.

## Hand-in and dates

| | |
|---|---|
| Presentations | Monday 30 November, 08:00–12:00, in class |
| Report and repository | Monday 30 November, 23:59, no late submission |
| Oral exam on the course material | Monday 7 December; hours and slots announced separately |

The report goes to DTU Learn as a PDF named `<project>-group<N>-report.pdf`. The
repository is private on GitHub, with `@licioromao` added as collaborator and the
submitted version tagged `v1.0`; a zip on DTU Learn is accepted if GitHub is not
workable for your group.

The journal PDFs in `papers/` are licensed to DTU. Use them inside the course and
do not redistribute them.

Questions: Licio Romao, `licio@dtu.dk`.
