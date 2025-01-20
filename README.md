# Balancing State Exploration and Skill Diversity in Unsupervised Skill Discovery

## Code
This article is currently under revision, and the code will be organized and open-sourced after acceptance.




## Skill Visualization


### ComSD (ours)
(balance state exploration and skill diversity)

ComSD can discover diverse behaviors at different activity levels, including both dynamic movements and static postures.


<table style="width: 100%;">
  <tr>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w12.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">Flip Forward</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w11.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">flip backward</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w10.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">failed flip</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w9.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">advance on knees</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w8.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">lie down & kick back</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w7.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">Crawl Forward</span>
    </td>
  </tr>
  <tr>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w6.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">描述 7</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w5.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">描述 8</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w4.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">描述 9</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w3.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">描述 10</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w2.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">描述 11</span>
    </td>
    <td style="width: 16.66%; text-align: center;">
      <img src="comsdgif/w1.gif" width="100" height="100"><br>
      <span style="font - size: 12px; text-align: center;">描述 12</span>
    </td>
  </tr>
</table>

|| | | | | |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 1](comsdgif/c12.gif)<br>描述 1 |![GIF 2](comsdgif/c11.gif)<br>描述 2 |![GIF 3](comsdgif/c10.gif)<br>描述 3 |![GIF 4](comsdgif/c9.gif)<br>描述 4 |![GIF 5](comsdgif/c8.gif)<br>描述 5 |![GIF 6](comsdgif/c7.gif)<br>描述 6 |
|![GIF 7](comsdgif/c6.gif)<br>描述 7 |![GIF 8](comsdgif/c5.gif)<br>描述 8 |![GIF 9](comsdgif/c4.gif)<br>描述 9 |![GIF 10](comsdgif/c3.gif)<br>描述 10 |![GIF 11](comsdgif/c2.gif)<br>描述 11 |![GIF 12](comsdgif/c1.gif)<br>描述 12 |


---
Recent advanced approaches cannot take both state exploration and skill diversity into account. They also can't generate behaviors at different activity levels.


### CIC (baseline)
(high state exploration, low skill diversity)

CIC is able to produce dynamic movements, but the generated skills are indistinguishable and homogeneous. It can't generate behaviors at other activity levels. 

|| | | | | |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 1](cic/w1.gif)<br>|![GIF 2](cic/w2.gif)<br>|![GIF 3](cic/w3.gif)<br> |![GIF 4](cic/c1.gif)<br> |![GIF 5](cic/c2.gif)<br>|![GIF 6](cic/c3.gif)<br> |


### APS (baseline)
(high skill diversity, low state exploration)

APS can generate non-homogeneous postures, but it suffers from lazy state exploration. It can't generate behaviors at other activity levels. 

|| | | | | |
| :---: | :---: | :---: | :---: | :---: | :---: |
|![GIF 1](aps/w1.gif)<br> |![GIF 2](aps/w2.gif)<br>|![GIF 3](aps/w3.gif)<br>|![GIF 4](aps/c1.gif)<br> |![GIF 5](aps/c2.gif)<br>|![GIF 6](aps/c3.gif)<br>|



