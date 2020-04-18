# Counterfactual Regret Minimization

Minimal implementation of CFR to learn Nash equilibrium strategies
for Kuhn poker and regret matching for rock paper scissors.

## Output

```
$ python cfr.py 
Finished training with an avg utility of -0.056543263715354064
------------
Completed 10 trials of 100000 games.
  Mean utility: -0.05474
  Std: 0.00043
------------
KL(sigma* | sigma)
  sigma(0): [0.77106378 0.22893622]: KL=0.00000
  sigma(1): [0.99999099 0.00000901]: KL=0.00001
  sigma(2): [0.31179841 0.68820159]: KL=0.00000
  sigma(0B): [0.9999985 0.0000015]: KL=0.00000
  sigma(0P): [0.66712108 0.33287892]: KL=0.00000
  sigma(1B): [0.66817037 0.33182963]: KL=0.00001
  sigma(1P): [0.99998949 0.00001051]: KL=0.00001
  sigma(2B): [0.0000015 0.9999985]: KL=0.00000
  sigma(2P): [0.0000015 0.9999985]: KL=0.00000
  sigma(0PB): [0.99999903 0.00000097]: KL=0.00000
  sigma(1PB): [0.43512826 0.56487174]: KL=0.00001
  sigma(2PB): [0.00000241 0.99999759]: KL=0.00000
```

## References

http://modelai.gettysburg.edu/2013/cfr/

https://en.wikipedia.org/wiki/Kuhn_poker
