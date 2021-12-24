# Notes about handing in job in Euler Cluster

Detailed tutorial can be found here: [https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#Using_the_batch_system](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters#Using_the_batch_system)
## Basic command
---
```
bsub -n 4-R "rusage[mem=4000,ngpus_excl_p=1]" -W 8:00 bash run_test.sh 
```
```-n 4``` is the number of cpu

```mem``` is memory per core (in mb)

```ngpus_excl_p``` is the number of gpu

```-I```  start the job interactively in the current terminal, but the output may show up after the whole program finish.

```-W HH:MM``` expected time to run the job, i.e. 20:00 means after 20 hours the job will be killed.

---
```
bbjobs
```
Giving details of the current running job

---
```
bkill <job-id>
```
kill the current running job, whose id is ```<job-id>```