# Common questions

## Why not R

R is a beautiful languages crafted by statisticians for statisticians. However, It doesn't scale as well as python (see [here](https://rstudio-pubs-static.s3.amazonaws.com/72295_692737b667614d369bd87cb0f51c9a4b.html)) and package environment reproducibility is not as complete as in Python or other languages. Thus, Python being a full-fledged programming language, it is production ready.

## What about Jupyter and notebooks

Notebooks are the entry points of most ML tutorials. However, it has many [drawbacks](https://www.youtube.com/watch?v=7jiPeIFXb6U)... Netflix went to great length to [use notebook in production](https://medium.com/netflix-techblog/notebook-innovation-591ee3221233). But you are not netflix. There is the same problem with R Shiny community: "Shiny apps are developedby R users, who aren't necessarily software engineers (Joe Cheng, RShiny creator[here](https://speakerdeck.com/jcheng5/shiny-in-production))

A fair learning would start by learning actual python, OOP and so on *before* using notebooks. Once you have a fair level in python and software engineering, there are two main use of notebooks : plot prototypes and reports. But that is a small part of the job, so not useful to cover it.

## Data scientists are not developer

You write python code. You're a developer. You learn and respect software engineering. Period.

Besides, buggy code is bade science, as [outlined](https://twitter.com/fchollet/status/1018396455533506560) by François Chollet:

> Buggy code is bad science. Poorly tuned benchmarks are bad science. Poorly factored code is bad science (hinders reproducibility, increases chances of a mistake). If your field is all about empirical validation, then your code *is* a large part of your scientific output.