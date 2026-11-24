# Every selected final result

All 210 rows, all four strengths and all three seeds. Seed 13 is tuning. Full precision and additional metrics are in [final-results-scalars.csv](final-results-scalars.csv). `P/C/J` is point / CE / joint screen; baseline is not compared to itself. nAUC is the strongest identifier-only attack.

## K = 2

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c2-baseline0-h1-s13 | tuning | 0.9758 | 0.8939 | 0.9976 | 0.0116 | 0.9750 | 249.4 | —/—/— |
| c2-baseline0-h1-s29 | fixed_configuration_replica | 0.9707 | 0.8535 | 0.9977 | 0.0119 | 0.9270 | 328.7 | —/—/— |
| c2-baseline0-h1-s47 | fixed_configuration_replica | 0.9707 | 0.8636 | 0.9978 | 0.0107 | 0.8670 | 265.1 | —/—/— |
| c2-control0-h1-s13 | tuning | 0.9747 | 0.8838 | 0.9973 | 0.0173 | 0.7380 | 330.6 | pass/fail/fail |
| c2-control0-h1-s29 | fixed_configuration_replica | 0.9778 | 0.8889 | 0.9978 | 0.0152 | 0.9400 | 819.3 | fail/fail/fail |
| c2-control0-h1-s47 | fixed_configuration_replica | 0.9636 | 0.8586 | 0.9974 | 0.0149 | 0.8790 | 401.5 | fail/fail/fail |
| c2-dp1-h3-s13 | tuning | 0.2303 | 0.0556 | 0.9627 | 0.2148 | 0.2630 | 2201.4 | fail/fail/fail |
| c2-dp1-h3-s29 | fixed_configuration_replica | 0.1909 | 0.0303 | 0.9602 | 0.2261 | 0.3540 | 2239.7 | fail/fail/fail |
| c2-dp1-h3-s47 | fixed_configuration_replica | 0.3374 | 0.0859 | 0.9670 | 0.2064 | 0.2340 | 2172.9 | fail/fail/fail |
| c2-dp2-h4-s13 | tuning | 0.5758 | 0.3889 | 0.9793 | 0.1280 | 0.2470 | 1646.4 | fail/fail/fail |
| c2-dp2-h4-s29 | fixed_configuration_replica | 0.5545 | 0.3333 | 0.9780 | 0.1316 | 0.3700 | 1651.5 | fail/fail/fail |
| c2-dp2-h4-s47 | fixed_configuration_replica | 0.5505 | 0.3687 | 0.9776 | 0.1342 | 0.2850 | 1650.9 | fail/fail/fail |
| c2-dp4-h4-s13 | tuning | 0.6838 | 0.5101 | 0.9833 | 0.0944 | 0.2150 | 1651.7 | fail/fail/fail |
| c2-dp4-h4-s29 | fixed_configuration_replica | 0.7040 | 0.5556 | 0.9852 | 0.0959 | 0.3380 | 1646.0 | fail/fail/fail |
| c2-dp4-h4-s47 | fixed_configuration_replica | 0.6909 | 0.5051 | 0.9829 | 0.0944 | 0.2320 | 1647.5 | fail/fail/fail |
| c2-dp8-h4-s13 | tuning | 0.7879 | 0.6263 | 0.9878 | 0.0654 | 0.3550 | 1649.1 | fail/fail/fail |
| c2-dp8-h4-s29 | fixed_configuration_replica | 0.7424 | 0.5657 | 0.9870 | 0.0779 | 0.3300 | 1651.1 | fail/fail/fail |
| c2-dp8-h4-s47 | fixed_configuration_replica | 0.7707 | 0.5960 | 0.9873 | 0.0733 | 0.2560 | 1652.0 | fail/fail/fail |
| c2-trap0.5-h4-s13 | tuning | 0.9889 | 0.9444 | 0.9975 | 0.0126 | 0.9650 | 295.8 | fail/fail/fail |
| c2-trap0.5-h4-s29 | fixed_configuration_replica | 0.9747 | 0.8990 | 0.9980 | 0.0116 | 0.9010 | 500.0 | fail/pass/fail |
| c2-trap0.5-h4-s47 | fixed_configuration_replica | 0.9778 | 0.9040 | 0.9978 | 0.0118 | 0.9940 | 516.0 | fail/fail/fail |
| c2-trap1-h1-s13 | tuning | 0.9818 | 0.9091 | 0.9978 | 0.0104 | 0.6980 | 474.0 | pass/pass/pass |
| c2-trap1-h1-s29 | fixed_configuration_replica | 0.9737 | 0.8788 | 0.9980 | 0.0104 | 0.3990 | 399.5 | pass/pass/pass |
| c2-trap1-h1-s47 | fixed_configuration_replica | 0.9768 | 0.8838 | 0.9979 | 0.0099 | 0.4670 | 621.9 | pass/pass/pass |
| c2-trap2-h4-s13 | tuning | 0.9808 | 0.9141 | 0.9974 | 0.0140 | 0.2480 | 524.4 | pass/fail/fail |
| c2-trap2-h4-s29 | fixed_configuration_replica | 0.9778 | 0.9091 | 0.9974 | 0.0142 | 0.3490 | 399.0 | pass/fail/fail |
| c2-trap2-h4-s47 | fixed_configuration_replica | 0.9727 | 0.8687 | 0.9970 | 0.0170 | 0.1020 | 220.2 | pass/fail/fail |
| c2-trap4-h1-s13 | tuning | 0.9657 | 0.8636 | 0.9970 | 0.0151 | 0.2560 | 374.6 | fail/fail/fail |
| c2-trap4-h1-s29 | fixed_configuration_replica | 0.9636 | 0.8434 | 0.9972 | 0.0127 | 0.2060 | 549.3 | pass/fail/fail |
| c2-trap4-h1-s47 | fixed_configuration_replica | 0.9707 | 0.8687 | 0.9971 | 0.0154 | 0.1720 | 273.0 | pass/fail/fail |

## K = 4

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c4-baseline0-h1-s13 | tuning | 0.9717 | 0.8737 | 0.9975 | 0.0123 | 0.9510 | 260.5 | —/—/— |
| c4-baseline0-h1-s29 | fixed_configuration_replica | 0.9626 | 0.8384 | 0.9976 | 0.0128 | 0.9220 | 334.6 | —/—/— |
| c4-baseline0-h1-s47 | fixed_configuration_replica | 0.9667 | 0.8737 | 0.9975 | 0.0112 | 0.7970 | 243.5 | —/—/— |
| c4-control0-h1-s13 | tuning | 0.9717 | 0.8737 | 0.9972 | 0.0154 | 0.8990 | 318.6 | pass/fail/fail |
| c4-control0-h1-s29 | fixed_configuration_replica | 0.9707 | 0.8990 | 0.9974 | 0.0159 | 0.7920 | 343.2 | pass/fail/fail |
| c4-control0-h1-s47 | fixed_configuration_replica | 0.9717 | 0.8788 | 0.9976 | 0.0150 | 0.6860 | 347.6 | pass/fail/fail |
| c4-dp1-h6-s13 | tuning | 0.0838 | 0.0000 | 0.9295 | 0.3428 | 0.2430 | 1056.0 | fail/fail/fail |
| c4-dp1-h6-s29 | fixed_configuration_replica | 0.0111 | 0.0000 | 0.9078 | 0.5114 | 0.3060 | 279.7 | fail/fail/fail |
| c4-dp1-h6-s47 | fixed_configuration_replica | 0.0626 | 0.0000 | 0.9224 | 0.3965 | 0.2280 | 597.5 | fail/fail/fail |
| c4-dp2-h3-s13 | tuning | 0.3465 | 0.1768 | 0.9723 | 0.1691 | 0.3800 | 1321.2 | fail/fail/fail |
| c4-dp2-h3-s29 | fixed_configuration_replica | 0.3364 | 0.1364 | 0.9709 | 0.1682 | 0.3090 | 1332.5 | fail/fail/fail |
| c4-dp2-h3-s47 | fixed_configuration_replica | 0.1929 | 0.0909 | 0.9697 | 0.1760 | 0.2250 | 1342.2 | fail/fail/fail |
| c4-dp4-h3-s13 | tuning | 0.5505 | 0.3586 | 0.9793 | 0.1213 | 0.2730 | 1327.7 | fail/fail/fail |
| c4-dp4-h3-s29 | fixed_configuration_replica | 0.5869 | 0.3737 | 0.9798 | 0.1200 | 0.3140 | 1326.8 | fail/fail/fail |
| c4-dp4-h3-s47 | fixed_configuration_replica | 0.5374 | 0.3889 | 0.9789 | 0.1232 | 0.2310 | 1326.8 | fail/fail/fail |
| c4-dp8-h6-s13 | tuning | 0.7172 | 0.5000 | 0.9838 | 0.1004 | 0.3230 | 1047.9 | fail/fail/fail |
| c4-dp8-h6-s29 | fixed_configuration_replica | 0.6768 | 0.3485 | 0.9812 | 0.1019 | 0.2910 | 600.6 | fail/fail/fail |
| c4-dp8-h6-s47 | fixed_configuration_replica | 0.6626 | 0.4091 | 0.9830 | 0.0921 | 0.2450 | 973.4 | fail/fail/fail |
| c4-trap0.5-h4-s13 | tuning | 0.9848 | 0.9343 | 0.9975 | 0.0125 | 0.8720 | 220.3 | pass/fail/fail |
| c4-trap0.5-h4-s29 | fixed_configuration_replica | 0.9778 | 0.8889 | 0.9977 | 0.0104 | 0.9070 | 360.1 | fail/pass/fail |
| c4-trap0.5-h4-s47 | fixed_configuration_replica | 0.9717 | 0.8687 | 0.9974 | 0.0108 | 0.8600 | 352.1 | fail/pass/fail |
| c4-trap1-h3-s13 | tuning | 0.9869 | 0.9444 | 0.9978 | 0.0133 | 0.7610 | 385.1 | pass/fail/fail |
| c4-trap1-h3-s29 | fixed_configuration_replica | 0.9707 | 0.8737 | 0.9975 | 0.0140 | 0.3880 | 327.5 | pass/fail/fail |
| c4-trap1-h3-s47 | fixed_configuration_replica | 0.9929 | 0.9646 | 0.9978 | 0.0163 | 0.2480 | 197.1 | pass/fail/fail |
| c4-trap2-h1-s13 | tuning | 0.9747 | 0.8889 | 0.9972 | 0.0143 | 0.1880 | 288.3 | pass/fail/fail |
| c4-trap2-h1-s29 | fixed_configuration_replica | 0.9596 | 0.8535 | 0.9974 | 0.0117 | 0.2500 | 401.2 | pass/pass/pass |
| c4-trap2-h1-s47 | fixed_configuration_replica | 0.9566 | 0.8384 | 0.9969 | 0.0149 | 0.1460 | 177.1 | fail/fail/fail |
| c4-trap4-h3-s13 | tuning | 0.9828 | 0.9192 | 0.9973 | 0.0183 | 0.1350 | 205.1 | pass/fail/fail |
| c4-trap4-h3-s29 | fixed_configuration_replica | 0.9677 | 0.8636 | 0.9969 | 0.0214 | 0.3790 | 166.8 | pass/fail/fail |
| c4-trap4-h3-s47 | fixed_configuration_replica | 0.9768 | 0.8889 | 0.9968 | 0.0191 | 0.2705 | 177.8 | pass/fail/fail |

## K = 6

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c6-baseline0-h3-s13 | tuning | 0.9798 | 0.9141 | 0.9974 | 0.0145 | 0.8860 | 255.4 | —/—/— |
| c6-baseline0-h3-s29 | fixed_configuration_replica | 0.9727 | 0.8889 | 0.9974 | 0.0150 | 0.7090 | 229.0 | —/—/— |
| c6-baseline0-h3-s47 | fixed_configuration_replica | 0.9758 | 0.8939 | 0.9971 | 0.0160 | 0.4510 | 171.0 | —/—/— |
| c6-control0-h1-s13 | tuning | 0.9505 | 0.8081 | 0.9968 | 0.0186 | 0.8600 | 475.4 | fail/fail/fail |
| c6-control0-h1-s29 | fixed_configuration_replica | 0.9667 | 0.8535 | 0.9968 | 0.0176 | 0.8230 | 574.6 | fail/fail/fail |
| c6-control0-h1-s47 | fixed_configuration_replica | 0.9687 | 0.8838 | 0.9970 | 0.0166 | 0.6990 | 433.1 | fail/fail/fail |
| c6-dp1-h6-s13 | tuning | 0.0788 | 0.0051 | 0.9391 | 0.3218 | 0.2140 | 1269.2 | fail/fail/fail |
| c6-dp1-h6-s29 | fixed_configuration_replica | 0.1071 | 0.0000 | 0.9423 | 0.3179 | 0.2860 | 1258.4 | fail/fail/fail |
| c6-dp1-h6-s47 | fixed_configuration_replica | 0.0374 | 0.0000 | 0.9375 | 0.3330 | 0.2740 | 1979.5 | fail/fail/fail |
| c6-dp2-h6-s13 | tuning | 0.2909 | 0.0556 | 0.9597 | 0.2153 | 0.1950 | 1274.9 | fail/fail/fail |
| c6-dp2-h6-s29 | fixed_configuration_replica | 0.2232 | 0.0404 | 0.9612 | 0.2065 | 0.4120 | 1269.3 | fail/fail/fail |
| c6-dp2-h6-s47 | fixed_configuration_replica | 0.2273 | 0.0404 | 0.9634 | 0.2028 | 0.2360 | 1269.4 | fail/fail/fail |
| c6-dp4-h6-s13 | tuning | 0.6051 | 0.3030 | 0.9760 | 0.1448 | 0.2495 | 1273.8 | fail/fail/fail |
| c6-dp4-h6-s29 | fixed_configuration_replica | 0.5576 | 0.2929 | 0.9761 | 0.1373 | 0.3130 | 1269.0 | fail/fail/fail |
| c6-dp4-h6-s47 | fixed_configuration_replica | 0.5899 | 0.2980 | 0.9773 | 0.1320 | 0.1810 | 1288.2 | fail/fail/fail |
| c6-dp8-h6-s13 | tuning | 0.7131 | 0.5253 | 0.9837 | 0.0908 | 0.4565 | 1277.4 | fail/fail/fail |
| c6-dp8-h6-s29 | fixed_configuration_replica | 0.6838 | 0.4798 | 0.9838 | 0.0935 | 0.2950 | 1277.8 | fail/fail/fail |
| c6-dp8-h6-s47 | fixed_configuration_replica | 0.6909 | 0.4040 | 0.9829 | 0.1051 | 0.3150 | 1263.0 | fail/fail/fail |
| c6-trap0.5-h3-s13 | tuning | 0.9859 | 0.9293 | 0.9980 | 0.0119 | 0.9400 | 379.7 | fail/pass/fail |
| c6-trap0.5-h3-s29 | fixed_configuration_replica | 0.9788 | 0.9040 | 0.9974 | 0.0130 | 0.6010 | 296.8 | pass/pass/pass |
| c6-trap0.5-h3-s47 | fixed_configuration_replica | 0.9687 | 0.8636 | 0.9974 | 0.0121 | 0.7890 | 394.1 | fail/pass/fail |
| c6-trap1-h4-s13 | tuning | 0.9848 | 0.9293 | 0.9975 | 0.0132 | 0.4500 | 275.7 | pass/pass/pass |
| c6-trap1-h4-s29 | fixed_configuration_replica | 0.9758 | 0.9040 | 0.9977 | 0.0114 | 0.3170 | 277.5 | pass/pass/pass |
| c6-trap1-h4-s47 | fixed_configuration_replica | 0.9798 | 0.9040 | 0.9972 | 0.0141 | 0.1820 | 234.4 | pass/pass/pass |
| c6-trap2-h4-s13 | tuning | 0.9717 | 0.8838 | 0.9971 | 0.0135 | 0.1960 | 342.4 | pass/pass/pass |
| c6-trap2-h4-s29 | fixed_configuration_replica | 0.9707 | 0.8939 | 0.9973 | 0.0135 | 0.3470 | 317.8 | pass/pass/pass |
| c6-trap2-h4-s47 | fixed_configuration_replica | 0.9657 | 0.8838 | 0.9973 | 0.0138 | 0.2270 | 278.5 | fail/pass/fail |
| c6-trap4-h4-s13 | tuning | 0.9626 | 0.8384 | 0.9967 | 0.0166 | 0.2080 | 414.3 | fail/fail/fail |
| c6-trap4-h4-s29 | fixed_configuration_replica | 0.9737 | 0.8889 | 0.9970 | 0.0164 | 0.3355 | 280.0 | pass/fail/fail |
| c6-trap4-h4-s47 | fixed_configuration_replica | 0.9667 | 0.8535 | 0.9973 | 0.0130 | 0.2570 | 575.5 | pass/pass/pass |

## K = 8

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c8-baseline0-h5-s13 | tuning | 0.9758 | 0.8788 | 0.9975 | 0.0117 | 0.9790 | 541.5 | —/—/— |
| c8-baseline0-h5-s29 | fixed_configuration_replica | 0.9859 | 0.9343 | 0.9974 | 0.0161 | 0.7500 | 247.4 | —/—/— |
| c8-baseline0-h5-s47 | fixed_configuration_replica | 0.9788 | 0.8939 | 0.9973 | 0.0161 | 0.7820 | 242.0 | —/—/— |
| c8-control0-h1-s13 | tuning | 0.9717 | 0.8990 | 0.9970 | 0.0181 | 0.6420 | 376.1 | pass/fail/fail |
| c8-control0-h1-s29 | fixed_configuration_replica | 0.9747 | 0.8939 | 0.9970 | 0.0195 | 0.5930 | 328.4 | fail/fail/fail |
| c8-control0-h1-s47 | fixed_configuration_replica | 0.9677 | 0.8838 | 0.9969 | 0.0198 | 0.2400 | 274.4 | fail/fail/fail |
| c8-dp1-h5-s13 | tuning | 0.0364 | 0.0000 | 0.9177 | 0.4311 | 0.2190 | 1432.5 | fail/fail/fail |
| c8-dp1-h5-s29 | fixed_configuration_replica | 0.0010 | 0.0000 | 0.8809 | 0.6205 | 0.4200 | 720.4 | fail/fail/fail |
| c8-dp1-h5-s47 | fixed_configuration_replica | 0.0030 | 0.0000 | 0.9015 | 0.5750 | 0.3255 | 701.6 | fail/fail/fail |
| c8-dp2-h6-s13 | tuning | 0.1636 | 0.0253 | 0.9616 | 0.2116 | 0.1830 | 1066.3 | fail/fail/fail |
| c8-dp2-h6-s29 | fixed_configuration_replica | 0.1768 | 0.0455 | 0.9630 | 0.2040 | 0.3350 | 1125.4 | fail/fail/fail |
| c8-dp2-h6-s47 | fixed_configuration_replica | 0.2071 | 0.0253 | 0.9586 | 0.2210 | 0.1850 | 1113.2 | fail/fail/fail |
| c8-dp4-h5-s13 | tuning | 0.4778 | 0.2121 | 0.9703 | 0.1727 | 0.2965 | 1447.9 | fail/fail/fail |
| c8-dp4-h5-s29 | fixed_configuration_replica | 0.4071 | 0.1465 | 0.9701 | 0.1814 | 0.4510 | 1465.3 | fail/fail/fail |
| c8-dp4-h5-s47 | fixed_configuration_replica | 0.2253 | 0.0556 | 0.9598 | 0.2258 | 0.2050 | 613.3 | fail/fail/fail |
| c8-dp8-h6-s13 | tuning | 0.6758 | 0.4747 | 0.9812 | 0.1115 | 0.2220 | 1133.0 | fail/fail/fail |
| c8-dp8-h6-s29 | fixed_configuration_replica | 0.6313 | 0.4343 | 0.9814 | 0.1061 | 0.2740 | 1112.7 | fail/fail/fail |
| c8-dp8-h6-s47 | fixed_configuration_replica | 0.6677 | 0.4495 | 0.9822 | 0.1024 | 0.2310 | 1103.9 | fail/fail/fail |
| c8-trap0.5-h4-s13 | tuning | 0.9788 | 0.9242 | 0.9973 | 0.0126 | 0.5050 | 359.8 | pass/fail/fail |
| c8-trap0.5-h4-s29 | fixed_configuration_replica | 0.9758 | 0.8838 | 0.9974 | 0.0132 | 0.8530 | 400.3 | fail/pass/fail |
| c8-trap0.5-h4-s47 | fixed_configuration_replica | 0.9768 | 0.9192 | 0.9974 | 0.0138 | 0.4220 | 277.3 | pass/pass/pass |
| c8-trap1-h3-s13 | tuning | 0.9859 | 0.9495 | 0.9977 | 0.0123 | 0.7450 | 318.3 | pass/fail/fail |
| c8-trap1-h3-s29 | fixed_configuration_replica | 0.9879 | 0.9444 | 0.9978 | 0.0120 | 0.3200 | 364.8 | pass/pass/pass |
| c8-trap1-h3-s47 | fixed_configuration_replica | 0.9768 | 0.8939 | 0.9968 | 0.0195 | 0.2760 | 176.2 | pass/fail/fail |
| c8-trap2-h3-s13 | tuning | 0.9808 | 0.9293 | 0.9975 | 0.0142 | 0.1990 | 270.5 | pass/fail/fail |
| c8-trap2-h3-s29 | fixed_configuration_replica | 0.9677 | 0.9091 | 0.9973 | 0.0157 | 0.3935 | 264.5 | fail/pass/fail |
| c8-trap2-h3-s47 | fixed_configuration_replica | 0.9747 | 0.8990 | 0.9974 | 0.0158 | 0.2110 | 296.4 | pass/pass/pass |
| c8-trap4-h3-s13 | tuning | 0.9717 | 0.9040 | 0.9972 | 0.0167 | 0.2930 | 298.3 | pass/fail/fail |
| c8-trap4-h3-s29 | fixed_configuration_replica | 0.9677 | 0.8990 | 0.9972 | 0.0189 | 0.4820 | 259.1 | fail/fail/fail |
| c8-trap4-h3-s47 | fixed_configuration_replica | 0.9707 | 0.8535 | 0.9971 | 0.0178 | 0.2470 | 257.0 | pass/fail/fail |

## K = 12

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c12-baseline0-h3-s13 | tuning | 0.9778 | 0.8990 | 0.9977 | 0.0116 | 0.7910 | 327.8 | —/—/— |
| c12-baseline0-h3-s29 | fixed_configuration_replica | 0.9747 | 0.8990 | 0.9974 | 0.0137 | 0.6410 | 286.7 | —/—/— |
| c12-baseline0-h3-s47 | fixed_configuration_replica | 0.9848 | 0.9293 | 0.9977 | 0.0119 | 0.8850 | 338.8 | —/—/— |
| c12-control0-h1-s13 | tuning | 0.9535 | 0.8737 | 0.9965 | 0.0192 | 0.3020 | 305.5 | fail/fail/fail |
| c12-control0-h1-s29 | fixed_configuration_replica | 0.9525 | 0.8586 | 0.9968 | 0.0189 | 0.6790 | 490.3 | fail/fail/fail |
| c12-control0-h1-s47 | fixed_configuration_replica | 0.9535 | 0.8535 | 0.9967 | 0.0192 | 0.3350 | 356.8 | fail/fail/fail |
| c12-dp1-h5-s13 | tuning | 0.0121 | 0.0000 | 0.9268 | 0.4052 | 0.2910 | 1481.0 | fail/fail/fail |
| c12-dp1-h5-s29 | fixed_configuration_replica | 0.0333 | 0.0000 | 0.9182 | 0.4544 | 0.3520 | 1514.3 | fail/fail/fail |
| c12-dp1-h5-s47 | fixed_configuration_replica | 0.0455 | 0.0000 | 0.9302 | 0.3833 | 0.2290 | 1493.4 | fail/fail/fail |
| c12-dp2-h5-s13 | tuning | 0.0970 | 0.0000 | 0.9498 | 0.2605 | 0.2090 | 1467.5 | fail/fail/fail |
| c12-dp2-h5-s29 | fixed_configuration_replica | 0.1485 | 0.0303 | 0.9571 | 0.2450 | 0.3190 | 1533.4 | fail/fail/fail |
| c12-dp2-h5-s47 | fixed_configuration_replica | 0.1758 | 0.0303 | 0.9560 | 0.2485 | 0.2590 | 1493.8 | fail/fail/fail |
| c12-dp4-h5-s13 | tuning | 0.4414 | 0.2020 | 0.9724 | 0.1626 | 0.2920 | 1465.4 | fail/fail/fail |
| c12-dp4-h5-s29 | fixed_configuration_replica | 0.3879 | 0.1667 | 0.9709 | 0.1723 | 0.3900 | 1536.8 | fail/fail/fail |
| c12-dp4-h5-s47 | fixed_configuration_replica | 0.3162 | 0.1515 | 0.9704 | 0.1793 | 0.2740 | 1514.1 | fail/fail/fail |
| c12-dp8-h5-s13 | tuning | 0.6182 | 0.3434 | 0.9799 | 0.1135 | 0.1710 | 1489.8 | fail/fail/fail |
| c12-dp8-h5-s29 | fixed_configuration_replica | 0.6768 | 0.4293 | 0.9807 | 0.1214 | 0.3330 | 1508.8 | fail/fail/fail |
| c12-dp8-h5-s47 | fixed_configuration_replica | 0.5556 | 0.2980 | 0.9777 | 0.1388 | 0.2520 | 1484.7 | fail/fail/fail |
| c12-trap0.5-h5-s13 | tuning | 0.9838 | 0.9394 | 0.9974 | 0.0142 | 0.7020 | 444.7 | pass/fail/fail |
| c12-trap0.5-h5-s29 | fixed_configuration_replica | 0.9606 | 0.8535 | 0.9969 | 0.0197 | 0.6580 | 336.8 | fail/fail/fail |
| c12-trap0.5-h5-s47 | fixed_configuration_replica | 0.9818 | 0.9293 | 0.9975 | 0.0145 | 0.6530 | 452.1 | pass/fail/fail |
| c12-trap1-h3-s13 | tuning | 0.9788 | 0.8990 | 0.9974 | 0.0118 | 0.6720 | 379.8 | pass/fail/fail |
| c12-trap1-h3-s29 | fixed_configuration_replica | 0.9727 | 0.8838 | 0.9974 | 0.0119 | 0.4300 | 349.4 | pass/pass/pass |
| c12-trap1-h3-s47 | fixed_configuration_replica | 0.9758 | 0.8990 | 0.9977 | 0.0113 | 0.2820 | 427.0 | pass/pass/pass |
| c12-trap2-h5-s13 | tuning | 0.9758 | 0.8990 | 0.9971 | 0.0155 | 0.2480 | 531.3 | pass/fail/fail |
| c12-trap2-h5-s29 | fixed_configuration_replica | 0.9859 | 0.9495 | 0.9975 | 0.0137 | 0.4980 | 524.1 | pass/pass/pass |
| c12-trap2-h5-s47 | fixed_configuration_replica | 0.9677 | 0.8838 | 0.9972 | 0.0164 | 0.1760 | 393.8 | fail/fail/fail |
| c12-trap4-h3-s13 | tuning | 0.9707 | 0.8838 | 0.9969 | 0.0152 | 0.2320 | 543.9 | pass/fail/fail |
| c12-trap4-h3-s29 | fixed_configuration_replica | 0.9778 | 0.9040 | 0.9974 | 0.0152 | 0.3890 | 325.7 | pass/fail/fail |
| c12-trap4-h3-s47 | fixed_configuration_replica | 0.9717 | 0.8889 | 0.9975 | 0.0131 | 0.3365 | 530.4 | fail/fail/fail |

## K = 16

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c16-baseline0-h5-s13 | tuning | 0.9737 | 0.8889 | 0.9975 | 0.0114 | 0.7750 | 508.8 | —/—/— |
| c16-baseline0-h5-s29 | fixed_configuration_replica | 0.9798 | 0.8990 | 0.9973 | 0.0131 | 0.6955 | 461.0 | —/—/— |
| c16-baseline0-h5-s47 | fixed_configuration_replica | 0.9636 | 0.8586 | 0.9972 | 0.0133 | 0.8000 | 432.3 | —/—/— |
| c16-control0-h1-s13 | tuning | 0.9333 | 0.8030 | 0.9962 | 0.0231 | 0.2310 | 333.1 | fail/fail/fail |
| c16-control0-h1-s29 | fixed_configuration_replica | 0.9475 | 0.8687 | 0.9964 | 0.0231 | 0.3880 | 353.0 | fail/fail/fail |
| c16-control0-h1-s47 | fixed_configuration_replica | 0.9485 | 0.8586 | 0.9966 | 0.0225 | 0.2580 | 401.2 | fail/fail/fail |
| c16-dp1-h5-s13 | tuning | 0.0101 | 0.0000 | 0.9340 | 0.4036 | 0.2180 | 1577.0 | fail/fail/fail |
| c16-dp1-h5-s29 | fixed_configuration_replica | 0.0374 | 0.0000 | 0.9335 | 0.3886 | 0.3420 | 1657.5 | fail/fail/fail |
| c16-dp1-h5-s47 | fixed_configuration_replica | 0.0152 | 0.0000 | 0.9268 | 0.4193 | 0.2200 | 1653.5 | fail/fail/fail |
| c16-dp2-h5-s13 | tuning | 0.0687 | 0.0000 | 0.9440 | 0.2914 | 0.1650 | 1646.0 | fail/fail/fail |
| c16-dp2-h5-s29 | fixed_configuration_replica | 0.1515 | 0.0051 | 0.9511 | 0.2666 | 0.3890 | 1664.2 | fail/fail/fail |
| c16-dp2-h5-s47 | fixed_configuration_replica | 0.0929 | 0.0051 | 0.9502 | 0.2691 | 0.2090 | 1627.6 | fail/fail/fail |
| c16-dp4-h5-s13 | tuning | 0.3182 | 0.1111 | 0.9697 | 0.1728 | 0.3510 | 1662.5 | fail/fail/fail |
| c16-dp4-h5-s29 | fixed_configuration_replica | 0.3323 | 0.1515 | 0.9686 | 0.1822 | 0.3230 | 1647.7 | fail/fail/fail |
| c16-dp4-h5-s47 | fixed_configuration_replica | 0.3172 | 0.1717 | 0.9721 | 0.1664 | 0.2400 | 1649.2 | fail/fail/fail |
| c16-dp8-h5-s13 | tuning | 0.5495 | 0.2980 | 0.9787 | 0.1294 | 0.2380 | 1650.3 | fail/fail/fail |
| c16-dp8-h5-s29 | fixed_configuration_replica | 0.5980 | 0.3838 | 0.9801 | 0.1251 | 0.3100 | 1669.6 | fail/fail/fail |
| c16-dp8-h5-s47 | fixed_configuration_replica | 0.5364 | 0.3333 | 0.9791 | 0.1256 | 0.1850 | 1670.7 | fail/fail/fail |
| c16-trap0.5-h3-s13 | tuning | 0.9737 | 0.8939 | 0.9974 | 0.0128 | 0.8670 | 393.5 | fail/fail/fail |
| c16-trap0.5-h3-s29 | fixed_configuration_replica | 0.9727 | 0.8687 | 0.9975 | 0.0117 | 0.8340 | 509.1 | fail/pass/fail |
| c16-trap0.5-h3-s47 | fixed_configuration_replica | 0.9657 | 0.8586 | 0.9974 | 0.0135 | 0.7025 | 505.7 | pass/pass/pass |
| c16-trap1-h5-s13 | tuning | 0.9778 | 0.8939 | 0.9974 | 0.0158 | 0.6250 | 433.5 | pass/fail/fail |
| c16-trap1-h5-s29 | fixed_configuration_replica | 0.9657 | 0.8939 | 0.9976 | 0.0116 | 0.5970 | 652.5 | fail/pass/fail |
| c16-trap1-h5-s47 | fixed_configuration_replica | 0.9727 | 0.9040 | 0.9976 | 0.0110 | 0.2690 | 581.5 | pass/pass/pass |
| c16-trap2-h3-s13 | tuning | 0.9737 | 0.8990 | 0.9973 | 0.0140 | 0.2740 | 470.8 | pass/fail/fail |
| c16-trap2-h3-s29 | fixed_configuration_replica | 0.9636 | 0.8788 | 0.9970 | 0.0166 | 0.2605 | 339.6 | fail/fail/fail |
| c16-trap2-h3-s47 | fixed_configuration_replica | 0.9646 | 0.8586 | 0.9970 | 0.0139 | 0.1910 | 429.5 | pass/fail/fail |
| c16-trap4-h3-s13 | tuning | 0.9556 | 0.8535 | 0.9969 | 0.0153 | 0.2680 | 420.7 | fail/fail/fail |
| c16-trap4-h3-s29 | fixed_configuration_replica | 0.9707 | 0.8889 | 0.9972 | 0.0147 | 0.2375 | 476.2 | pass/fail/fail |
| c16-trap4-h3-s47 | fixed_configuration_replica | 0.9657 | 0.8485 | 0.9971 | 0.0129 | 0.2520 | 748.5 | pass/pass/pass |

## K = 32

| Run | Seed role | Field acc. | Record acc. | Token acc. | CE | nAUC | Fit sec. | P/C/J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| c32-baseline0-h3-s13 | tuning | 0.9646 | 0.8434 | 0.9968 | 0.0159 | 0.8120 | 608.6 | —/—/— |
| c32-baseline0-h3-s29 | fixed_configuration_replica | 0.9697 | 0.8788 | 0.9972 | 0.0142 | 0.7040 | 754.3 | —/—/— |
| c32-baseline0-h3-s47 | fixed_configuration_replica | 0.9616 | 0.8232 | 0.9968 | 0.0150 | 0.6545 | 749.0 | —/—/— |
| c32-control0-h1-s13 | tuning | 0.9343 | 0.8182 | 0.9961 | 0.0254 | 0.2150 | 571.0 | fail/fail/fail |
| c32-control0-h1-s29 | fixed_configuration_replica | 0.9394 | 0.8333 | 0.9961 | 0.0257 | 0.3580 | 551.1 | fail/fail/fail |
| c32-control0-h1-s47 | fixed_configuration_replica | 0.9465 | 0.8586 | 0.9962 | 0.0256 | 0.2385 | 551.6 | fail/fail/fail |
| c32-dp1-h5-s13 | tuning | 0.0030 | 0.0000 | 0.9121 | 0.5255 | 0.1780 | 2036.0 | fail/fail/fail |
| c32-dp1-h5-s29 | fixed_configuration_replica | 0.0010 | 0.0000 | 0.9146 | 0.5071 | 0.3530 | 1936.6 | fail/fail/fail |
| c32-dp1-h5-s47 | fixed_configuration_replica | 0.0010 | 0.0000 | 0.9151 | 0.5162 | 0.2220 | 1908.7 | fail/fail/fail |
| c32-dp2-h5-s13 | tuning | 0.0071 | 0.0000 | 0.9420 | 0.3714 | 0.2100 | 2042.6 | fail/fail/fail |
| c32-dp2-h5-s29 | fixed_configuration_replica | 0.0030 | 0.0000 | 0.9411 | 0.3754 | 0.3650 | 2047.0 | fail/fail/fail |
| c32-dp2-h5-s47 | fixed_configuration_replica | 0.0051 | 0.0000 | 0.9354 | 0.3688 | 0.2810 | 2065.3 | fail/fail/fail |
| c32-dp4-h5-s13 | tuning | 0.0212 | 0.0000 | 0.9529 | 0.2565 | 0.2650 | 2029.9 | fail/fail/fail |
| c32-dp4-h5-s29 | fixed_configuration_replica | 0.0667 | 0.0000 | 0.9529 | 0.2504 | 0.2960 | 2024.9 | fail/fail/fail |
| c32-dp4-h5-s47 | fixed_configuration_replica | 0.0263 | 0.0000 | 0.9508 | 0.2643 | 0.2530 | 2018.4 | fail/fail/fail |
| c32-dp8-h5-s13 | tuning | 0.1091 | 0.0152 | 0.9615 | 0.1886 | 0.1800 | 2107.1 | fail/fail/fail |
| c32-dp8-h5-s29 | fixed_configuration_replica | 0.1030 | 0.0606 | 0.9661 | 0.1909 | 0.3270 | 2037.5 | fail/fail/fail |
| c32-dp8-h5-s47 | fixed_configuration_replica | 0.1879 | 0.0556 | 0.9652 | 0.1903 | 0.2310 | 2044.7 | fail/fail/fail |
| c32-trap0.5-h3-s13 | tuning | 0.9636 | 0.8485 | 0.9970 | 0.0142 | 0.6245 | 789.9 | pass/pass/pass |
| c32-trap0.5-h3-s29 | fixed_configuration_replica | 0.9616 | 0.8485 | 0.9969 | 0.0146 | 0.5150 | 537.1 | pass/fail/fail |
| c32-trap0.5-h3-s47 | fixed_configuration_replica | 0.9707 | 0.8687 | 0.9973 | 0.0147 | 0.8235 | 881.3 | fail/pass/fail |
| c32-trap1-h3-s13 | tuning | 0.9798 | 0.9192 | 0.9974 | 0.0142 | 0.3060 | 723.0 | pass/pass/pass |
| c32-trap1-h3-s29 | fixed_configuration_replica | 0.9667 | 0.8535 | 0.9969 | 0.0148 | 0.4500 | 555.8 | pass/fail/fail |
| c32-trap1-h3-s47 | fixed_configuration_replica | 0.9586 | 0.8333 | 0.9968 | 0.0161 | 0.1770 | 535.8 | pass/fail/fail |
| c32-trap2-h5-s13 | tuning | 0.9646 | 0.8939 | 0.9970 | 0.0139 | 0.1600 | 1009.4 | pass/pass/pass |
| c32-trap2-h5-s29 | fixed_configuration_replica | 0.9515 | 0.8889 | 0.9970 | 0.0168 | 0.2920 | 673.1 | fail/fail/fail |
| c32-trap2-h5-s47 | fixed_configuration_replica | 0.9697 | 0.8737 | 0.9969 | 0.0166 | 0.1305 | 602.3 | pass/fail/fail |
| c32-trap4-h3-s13 | tuning | 0.9677 | 0.9040 | 0.9970 | 0.0165 | 0.4185 | 692.1 | pass/fail/fail |
| c32-trap4-h3-s29 | fixed_configuration_replica | 0.9697 | 0.8788 | 0.9973 | 0.0155 | 0.3240 | 703.4 | pass/fail/fail |
| c32-trap4-h3-s47 | fixed_configuration_replica | 0.9596 | 0.8687 | 0.9971 | 0.0145 | 0.2560 | 1067.3 | pass/pass/pass |

