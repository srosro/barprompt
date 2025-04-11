# barprompt

An open CLI to connect Langfuse prompts to Promptfoo's Eval.  After creating an environment file with your OpenAI, Anthropic, etc API keys, you can run...

```
$ poetry python barprompt.py
> #####################################
> # Welcome to BARPROMPT
> # The app that integrates langfuse (prompt management) into promptfoo (prompt evaluation)
> #####################################
>  
> Please input the name of the prompt:
$ message-enricher

> Great! Please input the version:
$ 10

> Great! Please input the name of the dataset:
$ check_performance

> Great, we'll evaluate:
> - message-enricher (v10) with dataset check_performance, which will be 1,250 LLM queries. Continue (y/n):
$ y

> Experiment complete. You can find this experiment in https://padmin.codel.one/project/cm9aqr01i000mp907usz9ezvz/datasets/cm9cl3u4h0012p907ttsfx7ec
```

