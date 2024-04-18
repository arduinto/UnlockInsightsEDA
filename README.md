# UnlockInsightsEDA
##### __GOAL__
The main task is to use historical data about a customer, which is collected month-by-month (like their spending habits, payment patterns, credit card balances, etc.) to estimate the likelihood that they will fail to repay what they owe on their credit card in the future.

Here, the _"target binary variable"_ is what participants of the competition are trying to predict. It's binary because it has two possible values:
* "default" (customer didn't pay), or
* "no default" (customer paid).

The way this target is determined is by watching the customer's behavior for 18 months after their most recent credit card statement. If, during those 18 months of observation, the customer fails to make a payment within 120 days (4 months) from the date of their last statement, they are marked as having defaulted on their credit card balance. So, even if they make a payment on the 121st day or anytime after, for the purposes of this competition, they're categorized as having defaulted.

##### __Analysis of Credit Card Statements per Customer__
1. A significant majority, precisely 80%, of our customers have 13 statements.
2. The remaining 20% possess a varied number of statements, ranging from 1 to 12.

__Insight__: The model cannot handle this variation in statement counts because it cannot handle different input sizes for each customer. An alternative approach to address this variability might be to consider only the latest statement or compute an average across all statements for each customer. Some machine learning models can indeed handle variable input sizes (like recurrent neural networks for sequences), while others require fixed-size inputs.
### Raw Data

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_ID</th>
      <th>S_2</th>
      <th>P_2</th>
      <th>D_39</th>
      <th>B_1</th>
      <th>B_2</th>
      <th>R_1</th>
      <th>S_3</th>
      <th>D_41</th>
      <th>B_3</th>
      <th>...</th>
      <th>D_137</th>
      <th>D_138</th>
      <th>D_139</th>
      <th>D_140</th>
      <th>D_141</th>
      <th>D_142</th>
      <th>D_143</th>
      <th>D_144</th>
      <th>D_145</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000099d6bd597052cdcda90ffabf56573fe9d7c79be5f...</td>
      <td>2017-03-09</td>
      <td>0.938477</td>
      <td>0.001734</td>
      <td>0.008728</td>
      <td>1.006836</td>
      <td>0.009224</td>
      <td>0.124023</td>
      <td>0.008774</td>
      <td>0.004707</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.002426</td>
      <td>0.003706</td>
      <td>0.003819</td>
      <td>NaN</td>
      <td>0.000569</td>
      <td>0.000610</td>
      <td>0.002674</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0000099d6bd597052cdcda90ffabf56573fe9d7c79be5f...</td>
      <td>2017-04-07</td>
      <td>0.936523</td>
      <td>0.005775</td>
      <td>0.004925</td>
      <td>1.000977</td>
      <td>0.006153</td>
      <td>0.126709</td>
      <td>0.000798</td>
      <td>0.002714</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.003956</td>
      <td>0.003166</td>
      <td>0.005032</td>
      <td>NaN</td>
      <td>0.009575</td>
      <td>0.005493</td>
      <td>0.009216</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0000099d6bd597052cdcda90ffabf56573fe9d7c79be5f...</td>
      <td>2017-05-28</td>
      <td>0.954102</td>
      <td>0.091492</td>
      <td>0.021652</td>
      <td>1.009766</td>
      <td>0.006817</td>
      <td>0.123962</td>
      <td>0.007599</td>
      <td>0.009422</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.003269</td>
      <td>0.007328</td>
      <td>0.000427</td>
      <td>NaN</td>
      <td>0.003429</td>
      <td>0.006985</td>
      <td>0.002604</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0000099d6bd597052cdcda90ffabf56573fe9d7c79be5f...</td>
      <td>2017-06-13</td>
      <td>0.960449</td>
      <td>0.002455</td>
      <td>0.013687</td>
      <td>1.002930</td>
      <td>0.001372</td>
      <td>0.117188</td>
      <td>0.000685</td>
      <td>0.005531</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.006119</td>
      <td>0.004517</td>
      <td>0.003201</td>
      <td>NaN</td>
      <td>0.008423</td>
      <td>0.006527</td>
      <td>0.009598</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0000099d6bd597052cdcda90ffabf56573fe9d7c79be5f...</td>
      <td>2017-07-16</td>
      <td>0.947266</td>
      <td>0.002483</td>
      <td>0.015190</td>
      <td>1.000977</td>
      <td>0.007607</td>
      <td>0.117310</td>
      <td>0.004654</td>
      <td>0.009308</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.003672</td>
      <td>0.004944</td>
      <td>0.008888</td>
      <td>NaN</td>
      <td>0.001670</td>
      <td>0.008125</td>
      <td>0.009827</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5531446</th>
      <td>fffff1d38b785cef84adeace64f8f83db3a0c31e8d92ea...</td>
      <td>2017-11-05</td>
      <td>0.979492</td>
      <td>0.416016</td>
      <td>0.020813</td>
      <td>0.828125</td>
      <td>0.003487</td>
      <td>0.090759</td>
      <td>0.005341</td>
      <td>0.025146</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.006836</td>
      <td>0.003679</td>
      <td>0.000457</td>
      <td>NaN</td>
      <td>0.000906</td>
      <td>0.001497</td>
      <td>0.002775</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5531447</th>
      <td>fffff1d38b785cef84adeace64f8f83db3a0c31e8d92ea...</td>
      <td>2017-12-23</td>
      <td>0.984863</td>
      <td>0.296631</td>
      <td>0.007210</td>
      <td>0.812500</td>
      <td>0.005905</td>
      <td>0.079895</td>
      <td>0.002243</td>
      <td>0.023697</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.003309</td>
      <td>0.007095</td>
      <td>0.007858</td>
      <td>NaN</td>
      <td>0.002777</td>
      <td>0.008224</td>
      <td>0.008858</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5531448</th>
      <td>fffff1d38b785cef84adeace64f8f83db3a0c31e8d92ea...</td>
      <td>2018-01-06</td>
      <td>0.982910</td>
      <td>0.444092</td>
      <td>0.013153</td>
      <td>0.815430</td>
      <td>0.003456</td>
      <td>0.100525</td>
      <td>0.002111</td>
      <td>0.012344</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.009956</td>
      <td>0.009995</td>
      <td>0.001088</td>
      <td>NaN</td>
      <td>0.005692</td>
      <td>0.006775</td>
      <td>0.005566</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5531449</th>
      <td>fffff1d38b785cef84adeace64f8f83db3a0c31e8d92ea...</td>
      <td>2018-02-06</td>
      <td>0.969727</td>
      <td>0.442627</td>
      <td>0.009857</td>
      <td>1.003906</td>
      <td>0.005116</td>
      <td>0.101807</td>
      <td>0.009933</td>
      <td>0.008575</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.005543</td>
      <td>0.006565</td>
      <td>0.009880</td>
      <td>NaN</td>
      <td>0.008125</td>
      <td>0.001168</td>
      <td>0.003983</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5531450</th>
      <td>fffff1d38b785cef84adeace64f8f83db3a0c31e8d92ea...</td>
      <td>2018-03-14</td>
      <td>0.981934</td>
      <td>0.002474</td>
      <td>0.000077</td>
      <td>0.992676</td>
      <td>0.000809</td>
      <td>0.119141</td>
      <td>0.003286</td>
      <td>0.014091</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.007317</td>
      <td>0.002888</td>
      <td>0.006207</td>
      <td>NaN</td>
      <td>0.005112</td>
      <td>0.003183</td>
      <td>0.001914</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5531451 rows × 191 columns</p>
</div>

### __Analysis of Credit Card Statements per Customer__
<table id="T_73713">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_73713_level0_col0" class="col_heading level0 col0" >Nbr of Statements per customer</th>
      <th id="T_73713_level0_col1" class="col_heading level0 col1" >Nbr of customers</th>
      <th id="T_73713_level0_col2" class="col_heading level0 col2" >% of customers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_73713_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_73713_row0_col0" class="data row0 col0" >13</td>
      <td id="T_73713_row0_col1" class="data row0 col1" >386034</td>
      <td id="T_73713_row0_col2" class="data row0 col2" >84.120000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_73713_row1_col0" class="data row1 col0" >12</td>
      <td id="T_73713_row1_col1" class="data row1 col1" >10623</td>
      <td id="T_73713_row1_col2" class="data row1 col2" >2.310000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_73713_row2_col0" class="data row2 col0" >11</td>
      <td id="T_73713_row2_col1" class="data row2 col1" >5961</td>
      <td id="T_73713_row2_col2" class="data row2 col2" >1.300000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_73713_row3_col0" class="data row3 col0" >10</td>
      <td id="T_73713_row3_col1" class="data row3 col1" >6721</td>
      <td id="T_73713_row3_col2" class="data row3 col2" >1.460000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_73713_row4_col0" class="data row4 col0" >9</td>
      <td id="T_73713_row4_col1" class="data row4 col1" >6411</td>
      <td id="T_73713_row4_col2" class="data row4 col2" >1.400000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_73713_row5_col0" class="data row5 col0" >8</td>
      <td id="T_73713_row5_col1" class="data row5 col1" >6110</td>
      <td id="T_73713_row5_col2" class="data row5 col2" >1.330000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_73713_row6_col0" class="data row6 col0" >7</td>
      <td id="T_73713_row6_col1" class="data row6 col1" >5198</td>
      <td id="T_73713_row6_col2" class="data row6 col2" >1.130000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_73713_row7_col0" class="data row7 col0" >6</td>
      <td id="T_73713_row7_col1" class="data row7 col1" >5515</td>
      <td id="T_73713_row7_col2" class="data row7 col2" >1.200000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_73713_row8_col0" class="data row8 col0" >5</td>
      <td id="T_73713_row8_col1" class="data row8 col1" >4671</td>
      <td id="T_73713_row8_col2" class="data row8 col2" >1.020000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_73713_row9_col0" class="data row9 col0" >4</td>
      <td id="T_73713_row9_col1" class="data row9 col1" >4673</td>
      <td id="T_73713_row9_col2" class="data row9 col2" >1.020000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_73713_row10_col0" class="data row10 col0" >3</td>
      <td id="T_73713_row10_col1" class="data row10 col1" >5778</td>
      <td id="T_73713_row10_col2" class="data row10 col2" >1.260000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_73713_row11_col0" class="data row11 col0" >2</td>
      <td id="T_73713_row11_col1" class="data row11 col1" >6098</td>
      <td id="T_73713_row11_col2" class="data row11 col2" >1.330000</td>
    </tr>
    <tr>
      <th id="T_73713_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_73713_row12_col0" class="data row12 col0" >1</td>
      <td id="T_73713_row12_col1" class="data row12 col1" >5120</td>
      <td id="T_73713_row12_col2" class="data row12 col2" >1.120000</td>
    </tr>
  </tbody>
</table>

### Identified Missing Value

<table id="T_fc073">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_fc073_level0_col0" class="col_heading level0 col0" >Feature</th>
      <th id="T_fc073_level0_col1" class="col_heading level0 col1" >Number of Missing Values</th>
      <th id="T_fc073_level0_col2" class="col_heading level0 col2" >% Missing Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_fc073_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_fc073_row0_col0" class="data row0 col0" >D_87</td>
      <td id="T_fc073_row0_col1" class="data row0 col1" >5527586</td>
      <td id="T_fc073_row0_col2" class="data row0 col2" >99.930000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_fc073_row1_col0" class="data row1 col0" >D_88</td>
      <td id="T_fc073_row1_col1" class="data row1 col1" >5525447</td>
      <td id="T_fc073_row1_col2" class="data row1 col2" >99.890000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_fc073_row2_col0" class="data row2 col0" >D_108</td>
      <td id="T_fc073_row2_col1" class="data row2 col1" >5502513</td>
      <td id="T_fc073_row2_col2" class="data row2 col2" >99.480000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_fc073_row3_col0" class="data row3 col0" >D_111</td>
      <td id="T_fc073_row3_col1" class="data row3 col1" >5500117</td>
      <td id="T_fc073_row3_col2" class="data row3 col2" >99.430000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_fc073_row4_col0" class="data row4 col0" >D_110</td>
      <td id="T_fc073_row4_col1" class="data row4 col1" >5500117</td>
      <td id="T_fc073_row4_col2" class="data row4 col2" >99.430000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_fc073_row5_col0" class="data row5 col0" >B_39</td>
      <td id="T_fc073_row5_col1" class="data row5 col1" >5497819</td>
      <td id="T_fc073_row5_col2" class="data row5 col2" >99.390000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_fc073_row6_col0" class="data row6 col0" >D_73</td>
      <td id="T_fc073_row6_col1" class="data row6 col1" >5475595</td>
      <td id="T_fc073_row6_col2" class="data row6 col2" >98.990000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_fc073_row7_col0" class="data row7 col0" >B_42</td>
      <td id="T_fc073_row7_col1" class="data row7 col1" >5459973</td>
      <td id="T_fc073_row7_col2" class="data row7 col2" >98.710000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_fc073_row8_col0" class="data row8 col0" >D_136</td>
      <td id="T_fc073_row8_col1" class="data row8 col1" >5336752</td>
      <td id="T_fc073_row8_col2" class="data row8 col2" >96.480000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_fc073_row9_col0" class="data row9 col0" >D_138</td>
      <td id="T_fc073_row9_col1" class="data row9 col1" >5336752</td>
      <td id="T_fc073_row9_col2" class="data row9 col2" >96.480000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_fc073_row10_col0" class="data row10 col0" >D_137</td>
      <td id="T_fc073_row10_col1" class="data row10 col1" >5336752</td>
      <td id="T_fc073_row10_col2" class="data row10 col2" >96.480000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_fc073_row11_col0" class="data row11 col0" >D_135</td>
      <td id="T_fc073_row11_col1" class="data row11 col1" >5336752</td>
      <td id="T_fc073_row11_col2" class="data row11 col2" >96.480000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_fc073_row12_col0" class="data row12 col0" >D_134</td>
      <td id="T_fc073_row12_col1" class="data row12 col1" >5336752</td>
      <td id="T_fc073_row12_col2" class="data row12 col2" >96.480000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_fc073_row13_col0" class="data row13 col0" >R_9</td>
      <td id="T_fc073_row13_col1" class="data row13 col1" >5218918</td>
      <td id="T_fc073_row13_col2" class="data row13 col2" >94.350000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_fc073_row14_col0" class="data row14 col0" >B_29</td>
      <td id="T_fc073_row14_col1" class="data row14 col1" >5150035</td>
      <td id="T_fc073_row14_col2" class="data row14 col2" >93.100000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_fc073_row15_col0" class="data row15 col0" >D_106</td>
      <td id="T_fc073_row15_col1" class="data row15 col1" >4990102</td>
      <td id="T_fc073_row15_col2" class="data row15 col2" >90.210000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_fc073_row16_col0" class="data row16 col0" >D_132</td>
      <td id="T_fc073_row16_col1" class="data row16 col1" >4988874</td>
      <td id="T_fc073_row16_col2" class="data row16 col2" >90.190000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_fc073_row17_col0" class="data row17 col0" >D_49</td>
      <td id="T_fc073_row17_col1" class="data row17 col1" >4985917</td>
      <td id="T_fc073_row17_col2" class="data row17 col2" >90.140000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_fc073_row18_col0" class="data row18 col0" >R_26</td>
      <td id="T_fc073_row18_col1" class="data row18 col1" >4922146</td>
      <td id="T_fc073_row18_col2" class="data row18 col2" >88.980000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_fc073_row19_col0" class="data row19 col0" >D_76</td>
      <td id="T_fc073_row19_col1" class="data row19 col1" >4908954</td>
      <td id="T_fc073_row19_col2" class="data row19 col2" >88.750000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_fc073_row20_col0" class="data row20 col0" >D_66</td>
      <td id="T_fc073_row20_col1" class="data row20 col1" >4908097</td>
      <td id="T_fc073_row20_col2" class="data row20 col2" >88.730000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_fc073_row21_col0" class="data row21 col0" >D_42</td>
      <td id="T_fc073_row21_col1" class="data row21 col1" >4740137</td>
      <td id="T_fc073_row21_col2" class="data row21 col2" >85.690000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_fc073_row22_col0" class="data row22 col0" >D_142</td>
      <td id="T_fc073_row22_col1" class="data row22 col1" >4587043</td>
      <td id="T_fc073_row22_col2" class="data row22 col2" >82.930000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_fc073_row23_col0" class="data row23 col0" >D_53</td>
      <td id="T_fc073_row23_col1" class="data row23 col1" >4084585</td>
      <td id="T_fc073_row23_col2" class="data row23 col2" >73.840000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_fc073_row24_col0" class="data row24 col0" >D_82</td>
      <td id="T_fc073_row24_col1" class="data row24 col1" >4058614</td>
      <td id="T_fc073_row24_col2" class="data row24 col2" >73.370000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row25" class="row_heading level0 row25" >25</th>
      <td id="T_fc073_row25_col0" class="data row25 col0" >D_50</td>
      <td id="T_fc073_row25_col1" class="data row25 col1" >3142402</td>
      <td id="T_fc073_row25_col2" class="data row25 col2" >56.810000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row26" class="row_heading level0 row26" >26</th>
      <td id="T_fc073_row26_col0" class="data row26 col0" >B_17</td>
      <td id="T_fc073_row26_col1" class="data row26 col1" >3137598</td>
      <td id="T_fc073_row26_col2" class="data row26 col2" >56.720000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row27" class="row_heading level0 row27" >27</th>
      <td id="T_fc073_row27_col0" class="data row27 col0" >D_105</td>
      <td id="T_fc073_row27_col1" class="data row27 col1" >3021431</td>
      <td id="T_fc073_row27_col2" class="data row27 col2" >54.620000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row28" class="row_heading level0 row28" >28</th>
      <td id="T_fc073_row28_col0" class="data row28 col0" >D_56</td>
      <td id="T_fc073_row28_col1" class="data row28 col1" >2990943</td>
      <td id="T_fc073_row28_col2" class="data row28 col2" >54.070000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row29" class="row_heading level0 row29" >29</th>
      <td id="T_fc073_row29_col0" class="data row29 col0" >S_9</td>
      <td id="T_fc073_row29_col1" class="data row29 col1" >2933643</td>
      <td id="T_fc073_row29_col2" class="data row29 col2" >53.040000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row30" class="row_heading level0 row30" >30</th>
      <td id="T_fc073_row30_col0" class="data row30 col0" >D_77</td>
      <td id="T_fc073_row30_col1" class="data row30 col1" >2513912</td>
      <td id="T_fc073_row30_col2" class="data row30 col2" >45.450000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row31" class="row_heading level0 row31" >31</th>
      <td id="T_fc073_row31_col0" class="data row31 col0" >D_43</td>
      <td id="T_fc073_row31_col1" class="data row31 col1" >1658396</td>
      <td id="T_fc073_row31_col2" class="data row31 col2" >29.980000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row32" class="row_heading level0 row32" >32</th>
      <td id="T_fc073_row32_col0" class="data row32 col0" >S_27</td>
      <td id="T_fc073_row32_col1" class="data row32 col1" >1400935</td>
      <td id="T_fc073_row32_col2" class="data row32 col2" >25.330000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row33" class="row_heading level0 row33" >33</th>
      <td id="T_fc073_row33_col0" class="data row33 col0" >D_46</td>
      <td id="T_fc073_row33_col1" class="data row33 col1" >1211699</td>
      <td id="T_fc073_row33_col2" class="data row33 col2" >21.910000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row34" class="row_heading level0 row34" >34</th>
      <td id="T_fc073_row34_col0" class="data row34 col0" >S_7</td>
      <td id="T_fc073_row34_col1" class="data row34 col1" >1020544</td>
      <td id="T_fc073_row34_col2" class="data row34 col2" >18.450000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row35" class="row_heading level0 row35" >35</th>
      <td id="T_fc073_row35_col0" class="data row35 col0" >S_3</td>
      <td id="T_fc073_row35_col1" class="data row35 col1" >1020544</td>
      <td id="T_fc073_row35_col2" class="data row35 col2" >18.450000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row36" class="row_heading level0 row36" >36</th>
      <td id="T_fc073_row36_col0" class="data row36 col0" >D_62</td>
      <td id="T_fc073_row36_col1" class="data row36 col1" >758161</td>
      <td id="T_fc073_row36_col2" class="data row36 col2" >13.710000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row37" class="row_heading level0 row37" >37</th>
      <td id="T_fc073_row37_col0" class="data row37 col0" >D_48</td>
      <td id="T_fc073_row37_col1" class="data row37 col1" >718725</td>
      <td id="T_fc073_row37_col2" class="data row37 col2" >12.990000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row38" class="row_heading level0 row38" >38</th>
      <td id="T_fc073_row38_col0" class="data row38 col0" >D_61</td>
      <td id="T_fc073_row38_col1" class="data row38 col1" >598052</td>
      <td id="T_fc073_row38_col2" class="data row38 col2" >10.810000</td>
    </tr>
    <tr>
      <th id="T_fc073_level0_row39" class="row_heading level0 row39" >39</th>
      <td id="T_fc073_row39_col0" class="data row39 col0" >P_3</td>
      <td id="T_fc073_row39_col1" class="data row39 col1" >301492</td>
      <td id="T_fc073_row39_col2" class="data row39 col2" >5.450000</td>
    </tr>
  </tbody>
</table>

![FreqOfCustStatements](https://github.com/arduinto/UnlockInsightsEDA/assets/142419799/be508b0a-0a71-4369-8d0d-cad6d7618a05)

![image](https://github.com/arduinto/UnlockInsightsEDA/assets/142419799/d32842c1-aea0-460f-a1b2-3815d31a49c5)

![FeaturesDistr](https://github.com/arduinto/UnlockInsightsEDA/assets/142419799/719196d5-75e7-41e6-bd5b-ad11c604885a)
