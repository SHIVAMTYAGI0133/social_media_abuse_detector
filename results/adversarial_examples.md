# Adversarial Robustness Evaluation (20 Tricky Test Cases)

- **v1 Pipeline:** Legacy preprocessing (`normalize_obfuscation=False`) + frozen `models/v1/` classifier (`threshold=0.500`).
- **v2 Pipeline:** Upgraded de-obfuscation preprocessing (`normalize_obfuscation=True`) + `v2 (Balanced @0.391)`.

## Overall Accuracy on 20 Adversarial Cases

| Pipeline | Correct / Total | Pass Rate |
| :--- | :---: | :---: |
| **v1 Baseline** | 8 / 20 | 40.0% |
| **v2 (Balanced @0.391)** | 20 / 20 | 100.0% |

## Breakdown by Category

| Category | Total | v1 Pass | v2 Pass |
| :--- | :---: | :---: | :---: |
| Leetspeak | 5 | 0/5 | 5/5 |
| Symbol Masking | 5 | 4/5 | 5/5 |
| Stretched Letters | 5 | 0/5 | 5/5 |
| Safe Negation | 5 | 4/5 | 5/5 |

## Per-Example Results

| ID | Category | Input Comment | Expected | v1 Result | v2 Result | v2 Normalized Text |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| ADV-01 | Leetspeak | `y0u ar3 an 1d10t and 5tup1d 10s3r` | Abusive | Safe (30.6%) FAIL | Abusive (100.0%) PASS | `you are an idiot and stupid loser` |
| ADV-02 | Leetspeak | `k1ll y0urs3lf n0w` | Abusive | Safe (18.3%) FAIL | Abusive (88.6%) PASS | `kill yourself now` |
| ADV-03 | Leetspeak | `sh1t h3ad b1tch` | Abusive | Safe (18.3%) FAIL | Abusive (100.0%) PASS | `shit head bitch` |
| ADV-04 | Leetspeak | `g0 d13 tr4sh` | Abusive | Safe (18.3%) FAIL | Abusive (99.5%) PASS | `go die trash` |
| ADV-05 | Leetspeak | `y0u 5uck a55h0l3` | Abusive | Safe (18.3%) FAIL | Abusive (100.0%) PASS | `you suck asshole` |
| ADV-06 | Symbol Masking | `f*ck y0u and your sh!t edits` | Abusive | Safe (39.0%) FAIL | Abusive (100.0%) PASS | `fuck you and your shit edits` |
| ADV-07 | Symbol Masking | `you are a b!tch and a c*nt` | Abusive | Abusive (99.1%) PASS | Abusive (100.0%) PASS | `you are a bitch and a cunt` |
| ADV-08 | Symbol Masking | `i will k!ll you if you touch this` | Abusive | Abusive (55.6%) PASS | Abusive (89.7%) PASS | `i will kill you if you touch this` |
| ADV-09 | Symbol Masking | `shut up you @$$hole and d!e` | Abusive | Abusive (99.8%) PASS | Abusive (100.0%) PASS | `shut up you asshole and die` |
| ADV-10 | Symbol Masking | `total idi*t and pathet!c l*ser` | Abusive | Abusive (53.4%) PASS | Abusive (100.0%) PASS | `total idiot and pathetic loser` |
| ADV-11 | Stretched Letters | `youuu areee sooooo stuuuupid` | Abusive | Safe (18.3%) FAIL | Abusive (100.0%) PASS | `you are so stupid` |
| ADV-12 | Stretched Letters | `i haaaate youuu iddddiot` | Abusive | Safe (18.3%) FAIL | Abusive (100.0%) PASS | `i hate you idiot` |
| ADV-13 | Stretched Letters | `shuuuuttt uuuuppp looooser` | Abusive | Safe (18.3%) FAIL | Abusive (99.9%) PASS | `shut up loser` |
| ADV-14 | Stretched Letters | `kiiiillll youuuurself` | Abusive | Safe (18.3%) FAIL | Abusive (91.0%) PASS | `kill yourself` |
| ADV-15 | Stretched Letters | `disguuuusting trrrrash` | Abusive | Safe (18.3%) FAIL | Abusive (98.7%) PASS | `disgusting trash` |
| ADV-16 | Safe Negation | `this edit is not bad at all thank you` | Safe | Safe (3.7%) PASS | Safe (0.3%) PASS | `this edit is not bad at all thank you` |
| ADV-17 | Safe Negation | `i do not hate your work let us never argue` | Safe | Abusive (51.1%) FAIL | Safe (15.1%) PASS | `i do not hate your work let us never argue` |
| ADV-18 | Safe Negation | `no insults please we should never attack users` | Safe | Safe (5.3%) PASS | Safe (0.7%) PASS | `no insults please we should never attack users` |
| ADV-19 | Safe Negation | `there is no threat here just polite constructive feedback` | Safe | Safe (5.5%) PASS | Safe (0.1%) PASS | `there is no threat here just polite constructive feedback` |
| ADV-20 | Safe Negation | `never be rude and do not make personal attacks` | Safe | Safe (38.9%) PASS | Safe (14.4%) PASS | `never be rude and do not make personal attacks` |
