# barprompt

An open CLI to connect Langfuse prompts to Promptfoo's Eval.  After creating an environment file with your OpenAI, Anthropic, etc API keys, you can run...

```
$ python -m ./barprompt.py
> #####################################
> # Welcome to BARPROMPT
> # The app that integrates langfuse (prompt management) into promptfoo (prompt evaluation)
> #####################################
> 
> Langfuse API enviornment credentials do not exist or are not valid, using the Natan org's public read-only keys... 
> Please input the name of a prompt:
$ message-enricher-v2-all-afects

> Great, there are 10 versions please input a version number or hit (enter) for the latest version:
$ 10

> Great, please input the name of another prompt, or hit (enter) to list datasets:
$ message-enricher-v2-negative-afect-optimized

> Great, there are 5 versions please input a version number or hit (enter) for the latest version:
$ 3

> Great, please input the name of another prompt, or hit (enter) to list datasets:
$ (enter)

> Plese choose a dataset to evaluate those prompts against.  Options:
>
> 1. all-affects
> 2. negative-affects
$ 1

> Great, we'll evaluate:
> - message-enricher-v2-all-afects v10
> - message-enricher-v2-negative-afect-optimized v3
> against the all-affects dataset, which will be 1,250 LLM queries.  Do you want to:
> 
> 1. Create the promptfoo yaml file
> 2. Run promptfoo directly and view the results
$ 2
```

**(script creates the yaml file, runs promptfoo eval, and then runs promptfoo view)**
