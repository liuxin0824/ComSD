# Balancing State Exploration and Skill Diversity in Unsupervised Skill Discovery

## Code
This article is currently under revision, and the code will be organized and open-sourced after acceptance.




## Videos of the discovered unsupervised skills



**ComSD (ours)** can discover diverse robot behaviors at different exploratory levels, including different kinds of dynamic movements and static postures. **(balances state exploration and skill diversity)**

Rurrent advanced methods can either only learn different static postures **(low state exploration)** or only produce highly dynamic movements that are homogeneous **(low skill diversity)**.

We visualized some representative skills discovered by ComSD (ours) in Walker and Cheetah. We also visualize two representative baselines: CIC (low skill diversity) and APS (low state exploration), for comparison. 

### Walker

APS discover different static postures at low activity. CIC discover homogeneous dynamically flipping.  ComSD (ours) discover diverse behaviors at different exploratory levels. Move the mouse over eacn skill video to get its concrete description. 

#### ComSD (ours) 

| <img src="comsdgif/w12.gif" title="Flip forward"> | <img src="comsdgif/w11.gif" title="Flip backward"> | <img src="comsdgif/w10.gif" title="Flip failed"> | <img src="comsdgif/w9.gif" title="Advance on knees"> | <img src="comsdgif/w8.gif" title="Lie down and kick back"> | <img src="comsdgif/w7.gif" title="Crawl forward"> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="comsdgif/w6.gif" title="Shake the left leg"> | <img src="comsdgif/w5.gif" title="Stand up"> | <img src="comsdgif/w4.gif" title="Flip to yoga"> | <img src="comsdgif/w3.gif" title="Lift the right leg"> | <img src="comsdgif/w2.gif" title="Lift the left leg"> | <img src="comsdgif/w1.gif" title="Kneel down"> |

### Cheetah



| <img src="comsdgif/c12.gif" title="Flip backward"> | <img src="comsdgif/c11.gif" title="Jump"> | <img src="comsdgif/c10.gif" title="Flip forward"> | <img src="comsdgif/c9.gif" title="Walk on hands"> | <img src="comsdgif/c8.gif" title="Step back"> | <img src="comsdgif/c7.gif" title="Walk forward"> |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="comsdgif/c6.gif" title="Flip and shake hand"> | <img src="comsdgif/c5.gif" title="Leg taps floor"> | <img src="comsdgif/c4.gif" title="Leg taps floor(slow)"> | <img src="comsdgif/c3.gif" title="Hand taps floor"> | <img src="comsdgif/c2.gif" title="Posture 1"> | <img src="comsdgif/c1.gif" title="Posture 2"> |


___
Recent advanced approaches cannot take both state exploration and skill diversity into account. They also can't generate behaviors at different activity levels.


### CIC


CIC is able to produce dynamic movements, but the generated skills are **indistinguishable and homogeneous**.  In addition, it can't generate behaviors at other activity levels (e.g., postures).  **(high state exploration, low skill diversity)**




#### Left three: Walker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Right three: Cheetah

 | | | | | | |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 1](cic/w1.gif)<br>|![GIF 2](cic/w2.gif)<br>|![GIF 3](cic/w3.gif)<br> |![GIF 4](cic/c1.gif)<br> |![GIF 5](cic/c2.gif)<br>|![GIF 6](cic/c3.gif)<br> |

Skills discovered by CIC are all trying to dynamically flip.



### APS


APS can generate non-homogeneous postures, but it suffers from **lazy state exploration**. It can't generate behaviors at other activity levels (e.g., dynamic flip). **(high skill diversity, low state exploration)**



#### Left three: Walker &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Right three: Cheetah
|![GIF 1](aps/w1.gif)<br> |![GIF 2](aps/w2.gif)<br>|![GIF 3](aps/w3.gif)<br>|![GIF 4](aps/w4.gif)<br> |![GIF 5](aps/w5.gif)<br>|![GIF 6](aps/w6.gif)<br>|
| :---: | :---: | :---: | :---: | :---: | :---: |

|![GIF 1](aps/c1.gif)<br> |![GIF 2](aps/c2.gif)<br>|![GIF 3](aps/c3.gif)<br>|![GIF 4](aps/c4.gif)<br> |![GIF 5](aps/c5.gif)<br>|![GIF 6](aps/c6.gif)<br>|
| :---: | :---: | :---: | :---: | :---: | :---: |


Skills discovered by APS are all different static postures.

___

The quantitative analysis also verify that our ComSD enables better balance between diversity and exploration than advanced baselines.


![Example Image](quantitative.png)





