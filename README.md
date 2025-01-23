# Balancing State Exploration and Skill Diversity in Unsupervised Skill Discovery

## Code
This article is currently under revision, and the code will be organized and open-sourced after acceptance.




## Videos of the discovered unsupervised skills



ComSD (ours) can discover diverse robot behaviors at different exploratory levels, including different kinds of dynamic movements and static postures. **(balances state exploration and skill diversity)**

Rurrent advanced methods can either only learn different static postures **(low state exploration)** or only produce highly dynamic movements that are homogeneous **(low skill diversity)**.

We visualized some representative skills discovered by ComSD (ours) in Walker and Cheetah. We also visualize two representative baselines: CIC (low skill diversity) and APS (low state exploration), for comparison. You can refer to Section V-J in the paper for more detailed analysis.

___
### Walker


**ComSD (ours)**: Diverse behaviors at different exploratory levels. Hover over each video for description.

| <img src="comsdgif/w12.gif" title="Flip forward"> | <img src="comsdgif/w10.gif" title="Flip failed"> | <img src="comsdgif/w8.gif" title="Lie down and kick back"> | <img src="comsdgif/w5.gif" title="Stand up"> | <img src="comsdgif/w4.gif" title="Flip to yoga"> | <img src="comsdgif/w2.gif" title="Lift the left leg"> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="comsdgif/w11.gif" title="Flip backward"> | <img src="comsdgif/w9.gif" title="Advance on knees"> | <img src="comsdgif/w7.gif" title="Crawl forward"> | <img src="comsdgif/w13.gif" title="left leg taps floor"> | <img src="comsdgif/w1.gif" title="Kneel down"> | <img src="comsdgif/w3.gif" title="Lift the right leg"> | 

**APS**: Different postures at low activity. (low state exploration)

|![GIF 1](aps/w1.gif)<br> |![GIF 2](aps/w2.gif)<br>|![GIF 3](aps/w3.gif)<br>|![GIF 4](aps/w4.gif)<br> |![GIF 5](aps/w5.gif)<br>|![GIF 6](aps/w6.gif)<br>|
| :---: | :---: | :---: | :---: | :---: | :---: |


**CIC**: Homogeneous dynamically flipping. (low skill diversity)

|![GIF 1](cic/w1.gif)<br> |![GIF 2](cic/w2.gif)<br>|![GIF 3](cic/w3.gif)<br>|![GIF 4](cic/w4.gif)<br> |![GIF 5](cic/w5.gif)<br>|![GIF 6](cic/w6.gif)<br>|
| :---: | :---: | :---: | :---: | :---: | :---: |

___
### Cheetah

**ComSD (ours)**: Diverse behaviors at different exploratory levels. Hover over each video for description.

| <img src="comsdgif/c12.gif" title="Flip backward"> | <img src="comsdgif/c10.gif" title="Flip forward"> | <img src="comsdgif/c8.gif" title="Step back"> | <img src="comsdgif/c6.gif" title="Flip and shake hand"> | <img src="comsdgif/c3.gif" title="Hand taps floor"> | <img src="comsdgif/c2.gif" title="Posture 1"> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="comsdgif/c11.gif" title="Jump"> | <img src="comsdgif/c9.gif" title="Walk on hands"> | <img src="comsdgif/c7.gif" title="Walk forward"> | <img src="comsdgif/c5.gif" title="Leg taps floor (fast)"> |  <img src="comsdgif/c4.gif" title="Leg taps floor (slow)"> | <img src="comsdgif/c1.gif" title="Posture 2"> |


**APS**: Different postures at low activity. (low state exploration)

|![GIF 1](aps/c1.gif)<br> |![GIF 2](aps/c2.gif)<br>|![GIF 3](aps/c3.gif)<br>|![GIF 4](aps/c4.gif)<br> |![GIF 5](aps/c5.gif)<br>|![GIF 6](aps/c6.gif)<br>|
| :---: | :---: | :---: | :---: | :---: | :---: |


**CIC**: Homogeneous dynamically flipping. (low skill diversity)

|![GIF 1](cic/c1.gif)<br> |![GIF 2](cic/c2.gif)<br>|![GIF 3](cic/c3.gif)<br>|![GIF 4](cic/c4.gif)<br> |![GIF 5](cic/c5.gif)<br>|![GIF 6](cic/c6.gif)<br>|
| :---: | :---: | :---: | :---: | :---: | :---: |


___

The quantitative analysis also verify that our ComSD enables better balance between diversity and exploration than advanced baselines. You can refer to Section V-G&H in the paper for more detailed analysis.



![Example Image](quantitative.png)





