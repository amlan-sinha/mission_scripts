import sys, os, time

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np
import pandas as pd
from GSFC_reference_waypoints import get_post_deployment_epoch_in_MJS_and_state_in_EJ2000, get_post_flyby_epoch_in_MJS_and_state_in_EJ2000, get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000

import matplotlib.pyplot as plt
from lineplot import LinePlot
from support import html_colors

DAY2SEC, SEC2DAY, RAD2DEG, DEG2RAD = 86400., 1/86400., 180 / np.pi, np.pi / 180

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase I)
def p1_initial_guess_fixed_BC():

    return np.array([4.824828659490347e+05, 0.000000000000000e+00,
       0.000000000000000e+00, 1.746418434106965e+00,
       0.000000000000000e+00, 4.331765163514802e-01,
       1.747309380000000e+00, 0.000000000000000e+00,
       1.000000000000000e+00, 1.753622230000000e+00,
       0.000000000000000e+00, 7.436954650000001e-01,
       1.855217080000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       4.527762910000000e+00, 6.023392968055219e-02,
       5.764384790000000e+00, 4.564705820000000e+00,
       0.000000000000000e+00, 4.929825823569898e+00,
       2.722311631876848e+00, 5.192088562610340e-01,
       6.283185307179586e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       5.902831210000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 4.629212180000000e+00,
       0.000000000000000e+00, 1.492863604704338e+00], dtype=np.longdouble)


# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase I)
def p1_initial_guess_other_BC():

    return np.array([4.502493938213671e+05, 0.000000000000000e+00,
       0.000000000000000e+00, 1.017238449737191e+00,
       5.580414883585732e+00, 5.583187679869422e-01,
       4.555661116413185e+00, 2.973649804361089e+00,
       1.000000000000000e+00, 4.852232615388868e+00,
       2.955753129459565e+00, 1.000000000000000e+00,
       3.465947676473578e-04, 3.067796358216713e+00,
       9.976537546258777e-01, 6.183582327869987e+00,
       0.000000000000000e+00, 5.419654923028792e-04,
       0.000000000000000e+00, 6.283185307179586e+00,
       7.021913616382374e-03, 5.415748347524338e+00,
       3.036391756493991e+00, 9.936234178612346e-01,
       9.163220290764947e-03, 3.120296860921324e+00,
       9.977836109863358e-01, 0.000000000000000e+00,
       0.000000000000000e+00, 2.101052143183159e-03,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.213180562898826e-03, 1.484686126911829e+00], dtype=np.longdouble)


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase II)
def p2_initial_guess_fixed_BC():

    return np.array([1.553216436488187e+07, 0.000000000000000e+00,
       0.000000000000000e+00, 5.067104200000000e+00,
       5.449956980000000e+00, 0.000000000000000e+00,
       6.283185307179586e+00, 3.216379820000000e+00,
       2.913733722539787e-02, 3.976057950000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       3.796834390000000e+00, 5.430800300000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.985674020000000e-01, 0.000000000000000e+00,
       0.000000000000000e+00, 5.303088860000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       4.196363110000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 3.707990260000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.704550480000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 2.870815240000000e+00,
       1.431982380000000e+00, 0.000000000000000e+00,
       3.082088090000000e+00, 5.448634480000000e-01,
       0.000000000000000e+00, 2.422919460000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 6.152207440000000e+00,
       3.065982640000000e+00, 0.000000000000000e+00,
       6.213620820000000e+00, 2.988454220000000e+00,
       0.000000000000000e+00, 6.158341440000000e+00,
       2.957302070000000e+00, 0.000000000000000e+00,
       2.959649760000000e+00, 1.590748790000000e-01,
       1.000000000000000e+00, 3.193319990000000e-02,
       6.283185307179586e+00, 0.000000000000000e+00,
       3.330967260000000e+00, 1.180521190000000e+00,
       0.000000000000000e+00, 2.969949060000000e+00,
       1.760155500000000e-01, 1.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 2.977747310000000e+00,
       6.283185307179586e+00, 1.000000000000000e+00,
       6.123307830000000e+00, 2.950476610000000e+00,
       1.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       3.993270950000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.933174990000000e+00, 9.798121234135689e-01,
       2.996253420000000e+00, 2.113666750000000e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       3.020204400000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 2.809102620000000e+00,
       0.000000000000000e+00, 2.917067630000000e+00,
       2.836805150000000e-01, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.952914400000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 2.981685010000000e+00,
       2.379516590000000e-01, 0.000000000000000e+00,
       0.000000000000000e+00, 1.311807440000000e-01,
       0.000000000000000e+00, 6.283185307179586e+00,
       3.379385230000000e+00, 0.000000000000000e+00,
       2.936188510000000e+00, 2.474369860000000e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.940768020000000e+00, 0.000000000000000e+00,
       6.283185307179586e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 4.346369800000000e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 4.276539370000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       8.275414500000000e-01, 0.000000000000000e+00,
       6.283185307179586e+00, 4.202300170000000e+00,
       0.000000000000000e+00, 5.133843860000000e+00,
       6.283185307179586e+00, 0.000000000000000e+00,
       5.355627390000000e+00, 4.894968970000000e-02,
       0.000000000000000e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 3.500667060000000e+00,
       5.727582290000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 3.794201150000000e+00,
       0.000000000000000e+00, 1.947757040000000e+00,
       5.125803250000000e+00, 0.000000000000000e+00,
       2.011793080000000e+00, 3.216763990000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       3.832141350000001e+00, 4.124051219678520e-02,
       4.217852460000000e+00, 1.084520550000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       6.283185307179586e+00, 0.000000000000000e+00,
       6.283185307179586e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 1.394051170000000e+00,
       6.478557100000000e-01, 0.000000000000000e+00,
       6.283185307179586e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 3.140586230000000e+00,
       5.722874170000000e+00, 0.000000000000000e+00,
       1.118625270000000e+00, 3.836337650000000e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       5.110998490000000e+00, 0.000000000000000e+00,
       4.351296920000000e+00, 2.864389940000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       3.383151470000000e+00, 0.000000000000000e+00,
       3.531450070000000e+00, 2.216517670000000e+00,
       0.000000000000000e+00, 4.168067460000000e+00,
       2.866360916544391e+00, 1.000000000000000e+00,
       9.740019480000001e-01, 6.283185307179586e+00,
       1.000000000000000e+00, 9.262198940000002e-01,
       2.608049780000000e-01, 1.000000000000000e+00,
       1.483430330000000e+00, 7.164955790000000e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       5.006145570000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       1.866617480000000e-01, 0.000000000000000e+00,
       7.442274010000000e-01, 2.302275350000000e-01,
       1.000000000000000e+00, 0.000000000000000e+00,
       4.153743840000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 9.449652050000001e-01,
       0.000000000000000e+00, 6.675711040000000e-01,
       2.185416427743711e-01, 1.000000000000000e+00,
       6.283185307179586e+00, 2.742101290000000e-01,
       0.000000000000000e+00, 6.261067517430549e-01,
       6.283185307179586e+00, 1.000000000000000e+00,
       6.283185307179586e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 2.200598060000000e+00,
       0.000000000000000e+00, 3.703787620000000e+00,
       2.926408370000000e+00, 1.000000000000000e+00,
       5.498286510000000e-01, 2.178707370000000e-01,
       3.683072126837780e-01, 2.757930270000000e+00,
       6.283185307179586e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 1.527689980000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       5.614570430000000e+00, 0.000000000000000e+00,
       6.283185307179586e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 8.694460070000001e-01,
       4.007144130000000e-01, 0.000000000000000e+00,
       5.892153470000000e+00, 1.046770450000000e-01,
       0.000000000000000e+00, 2.641019770000000e-01,
       6.283185307179586e+00, 0.000000000000000e+00,
       4.349015050000000e+00, 2.734512320000000e+00,
       0.000000000000000e+00, 5.711218710000000e-01,
       6.283185307179586e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 6.283185307179586e+00,
       0.000000000000000e+00, 1.111986840000000e-01,
       6.283185307179586e+00, 0.000000000000000e+00,
       1.422208630000000e+00, 1.335179870000000e+00,
       0.000000000000000e+00, 2.471803170000000e+00,
       5.190993380000000e+00, 0.000000000000000e+00,
       5.285894290000000e+00, 2.887225040000000e+00,
       0.000000000000000e+00, 1.681994270000000e+00,
       2.011649900000000e+00, 0.000000000000000e+00,
       6.022687400000000e+00, 3.133394450000000e+00,
       0.000000000000000e+00, 1.375854231219523e+00], dtype=np.longdouble)


# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase II)
def p2_initial_guess_other_BC():

    return np.array([1.539539586728636e+07, 0.000000000000000e+00,
       0.000000000000000e+00, 8.849861088218310e-03,
       2.481989964316016e-03, 1.312331918554677e-02,
       2.087189926323989e-02, 3.188337405302855e-02,
       1.445548123617731e-01, 4.474997086940683e-03,
       3.025065180894924e-03, 9.547426461270606e-03,
       2.175010971779960e+00, 4.136200651308372e+00,
       5.796672610210561e-01, 4.653796917656941e+00,
       7.397570233865397e-02, 7.978706384415972e-01,
       4.611892872999658e-03, 4.034726890334901e-03,
       1.649740077839628e-02, 2.675717760030499e+00,
       0.000000000000000e+00, 7.852163712517356e-03,
       3.598743223228568e-01, 0.000000000000000e+00,
       1.257446620190784e-02, 0.000000000000000e+00,
       3.537176981322024e+00, 1.020208762054606e-02,
       8.043594829987759e-02, 9.390219459938150e-02,
       8.671462352248921e-01, 0.000000000000000e+00,
       0.000000000000000e+00, 1.093624924600673e-02,
       4.972403446230865e+00, 0.000000000000000e+00,
       1.095291432560140e-04, 4.767268286781032e+00,
       5.696583588074738e+00, 3.367963610642256e-01,
       6.279035729364224e+00, 8.576556393506203e-02,
       9.791867227176846e-01, 5.220363670721675e+00,
       3.505395331166373e+00, 1.690022678913942e-01,
       3.233939270644896e+00, 1.195873477569424e-01,
       9.939275458750643e-01, 1.788844851676898e+00,
       0.000000000000000e+00, 1.109348050608815e-02,
       0.000000000000000e+00, 6.418795039434974e-01,
       0.000000000000000e+00, 6.297279489280337e-01,
       4.926510442848123e-02, 9.906884011621339e-01,
       2.283811512636505e+00, 1.766137285107033e-01,
       1.001761708497411e-02, 1.420636197471846e+00,
       3.238953648207918e-02, 9.995954679949206e-01,
       7.296089842630090e-01, 2.605790473413910e-02,
       9.901206275275505e-01, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       5.915385743651452e+00, 3.101837383726087e+00,
       2.269535788729439e-02, 9.548577139097658e-02,
       1.688344870658371e-03, 1.369227492567967e-01,
       1.418325492298351e+00, 0.000000000000000e+00,
       1.150645913050347e-02, 2.727044594130855e+00,
       5.682229687402327e+00, 9.907936346024356e-01,
       2.727602373366359e+00, 1.659430380886691e-02,
       9.645108583247105e-01, 4.160275221101905e-01,
       1.074541356796272e-02, 5.770530927790579e-01,
       0.000000000000000e+00, 4.142714879162745e+00,
       1.313738302853021e-01, 0.000000000000000e+00,
       5.413827703403013e+00, 2.311475003564099e-03,
       2.608958710630984e-01, 5.486692416956518e+00,
       2.933650001488089e-03, 2.792046726467456e+00,
       4.742533130089520e+00, 1.344490874499235e-02,
       0.000000000000000e+00, 0.000000000000000e+00,
       4.113988431133486e-03, 2.920119360618916e+00,
       4.227933074480691e-02, 9.951989482541960e-01,
       3.849986639439985e-01, 2.516648552023084e-02,
       6.776228563923579e-01, 1.810612821874287e+00,
       0.000000000000000e+00, 2.093910731966404e-02,
       0.000000000000000e+00, 2.102414970540268e+00,
       2.288669774325582e-02, 9.166690931251740e-01,
       4.377342263285142e+00, 5.461811717094759e-01,
       6.083221872665352e+00, 2.876649835394700e+00,
       9.916655526511906e-01, 2.861522494621492e-01,
       6.218275003520478e+00, 7.468281469010833e-01,
       5.568285836016361e+00, 2.658169636990967e-01,
       5.396809735828896e-01, 2.330777529119609e+00,
       2.207774923510222e+00, 1.315319872257292e-03,
       0.000000000000000e+00, 4.884360659169675e-01,
       6.496985685736526e-03, 3.059290112333984e+00,
       3.335922121831116e-01, 5.365979051245621e-01,
       5.314601460895261e+00, 2.782572807543562e+00,
       3.423525011697108e-02, 0.000000000000000e+00,
       0.000000000000000e+00, 9.471586707210657e-03,
       0.000000000000000e+00, 2.629486137130383e+00,
       5.994209832540097e-01, 4.457072453586532e-01,
       4.325784108191396e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 5.968898002548972e+00,
       1.035376804165909e-02, 0.000000000000000e+00,
       4.308441641533763e+00, 5.445725687574759e-03,
       6.283185307179586e+00, 2.851032592836982e+00,
       0.000000000000000e+00, 6.253184535291526e+00,
       6.268826118701178e+00, 3.882792858897407e-02,
       8.989510539126584e-01, 1.992396132149466e-01,
       9.865125314065160e-01, 8.500535396836578e-01,
       4.164038033496729e-02, 9.895952112669790e-01,
       4.676311912097243e-03, 6.283185307179586e+00,
       3.416178044020753e-02, 5.338214607046740e-03,
       0.000000000000000e+00, 2.622611588521337e-02,
       1.173792032983030e+00, 6.128819512531126e+00,
       9.958426960397971e-01, 2.850179498931661e+00,
       6.164172123173095e+00, 9.707443536603829e-01,
       5.435724432998263e+00, 0.000000000000000e+00,
       1.467219326089630e-03, 6.283185307179586e+00,
       6.283185307179586e+00, 1.061107448828953e-02,
       0.000000000000000e+00, 0.000000000000000e+00,
       1.165473263912169e-02, 0.000000000000000e+00,
       0.000000000000000e+00, 1.389934479885746e-02,
       1.229692333824270e+00, 6.283185307179586e+00,
       9.968959935135387e-01, 1.235782652530614e+00,
       2.276764971102491e-01, 9.977399973432458e-01,
       4.303215207959488e+00, 2.848064695077962e+00,
       1.000000000000000e+00, 1.143721855420234e+00,
       6.283185307179586e+00, 9.987306323271877e-01,
       1.140791679539658e+00, 4.596568041738389e-01,
       1.000000000000000e+00, 1.936250604426745e+00,
       3.392507728021221e+00, 1.328263115155755e-03,
       1.408299592195129e-02, 0.000000000000000e+00,
       3.269018692722313e-02, 1.352235417455356e-02,
       0.000000000000000e+00, 3.356769889491856e-02,
       4.704848143855723e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 1.154949104852490e-02,
       0.000000000000000e+00, 3.362879681470986e-02,
       6.283185307179586e+00, 7.795319206021721e-01,
       1.000000000000000e+00, 4.239778462086939e-01,
       4.647914093782526e-01, 4.895662791075193e-01,
       3.381289369988723e-01, 2.237560564658875e+00,
       4.467905550154204e-03, 0.000000000000000e+00,
       0.000000000000000e+00, 2.822724099857087e-02,
       0.000000000000000e+00, 0.000000000000000e+00,
       2.601344472459694e-02, 0.000000000000000e+00,
       0.000000000000000e+00, 2.354601063029198e-02,
       5.433622526931153e+00, 0.000000000000000e+00,
       3.544551429925954e-03, 1.195800233911284e+00,
       6.264766930877064e+00, 1.000000000000000e+00,
       3.806270384062364e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 1.194668103995653e-01,
       6.270175114754668e+00, 4.628267388515329e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       8.550747069188231e-03, 4.454276059345488e+00,
       1.734134346781061e+00, 4.038596226044298e-02,
       3.919935587337179e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       5.213323361100922e+00, 5.627526113709585e-03,
       0.000000000000000e+00, 2.153533733495510e-01,
       9.138905672518792e-03, 3.205758788949745e+00,
       2.702430652935425e+00, 8.368268452831170e-03,
       0.000000000000000e+00, 0.000000000000000e+00,
       1.005293490087638e-02, 6.283185307179586e+00,
       0.000000000000000e+00, 1.020957551887857e-02,
       3.996767399283897e+00, 5.770175520799954e-01,
       3.193833990851512e-02, 0.000000000000000e+00,
       2.123934484571427e+00, 9.910969374543374e-01,
       2.931597718023541e+00, 9.577581897129139e-01,
       4.462760381106934e-02, 0.000000000000000e+00,
       0.000000000000000e+00, 1.054083735331194e-02,
       6.283185307179586e+00, 0.000000000000000e+00,
       1.055510660502649e-02, 6.246643004517879e+00,
       2.284404318949895e+00, 9.901781842827176e-01,
       0.000000000000000e+00, 0.000000000000000e+00,
       1.048117802609041e-02, 2.313792143210109e+00,
       5.271685600122437e+00, 0.000000000000000e+00,
       2.644665006535815e+00, 4.840657391804597e+00,
       1.139073545558409e-03, 1.192188797496541e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00,
       0.000000000000000e+00, 0.000000000000000e+00], dtype=np.longdouble)


# get the throttle history from the control input history
def get_throttle_history(control_states):

    assert isinstance(control_states, np.ndarray)
    assert control_states.ndim == 2
    assert control_states.shape[1] == 3

    throttle_history = np.zeros(control_states.shape[0],)
    for i, control in enumerate(control_states):
        throttle_history[i] = np.linalg.norm(control)

    return throttle_history


# print control input history to screen
def output_control_to_screen(control):
    print('\nLength of control vector:', len(control), flush=True)
    print('\nThe control vector:', flush=True)
    for entry in control:
        print('{},'.format(entry), flush=True)


# integrate natural dynamics forward in time
def get_plots_of_phase_integration(body: pydylan.Body, eom: pydylan.eom.Ephemeris_nBP) -> (LinePlot, LinePlot):

    rk54 = pydylan.integrators.RK54()
    rk54.set_eom(eom)

    p = LinePlot()
    p.grid()
    p.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    p.set_ylabel(ylabel_in=r'Y (km)', fontsize=15)

    q = LinePlot()
    q.grid()
    q.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    q.set_ylabel(ylabel_in=r'Z (km)', fontsize=15)

    epoch, state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    if body.name=='Moon': epoch, state = get_rv_state_relative_to_moon(epoch, state)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    post_flyby_epoch, state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    if body.name=='Moon': post_flyby_epoch, state = get_rv_state_relative_to_moon(post_flyby_epoch, state)
    
    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    NRHO_insertion_epoch, state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()
    if body.name=='Moon': NRHO_insertion_epoch, state = get_rv_state_relative_to_moon(NRHO_insertion_epoch, state)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    number_of_steps = 5000
    number_of_spline_points = int(number_of_steps / 5)
    start_time, end_time = epoch, NRHO_insertion_epoch + 7 * DAY2SEC
    time_points = np.linspace(start_time, end_time, num=number_of_steps)

    splined_moon = pydylan.Body("Moon", True, number_of_spline_points, start_time, end_time)

    if body==pydylan.Body("Earth"):
        moon_states = np.zeros((len(time_points), 6))
        for i, time in enumerate(time_points):
            moon_states[i] = splined_moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(time)

        p.plot(xdata=moon_states[:, 0], ydata=moon_states[:, 1], color='black', alpha=0.3, linestyle='--')
        q.plot(xdata=moon_states[:, 0], ydata=moon_states[:, 2], color='black', alpha=0.3, linestyle='--')
    else:
        earth_states = np.zeros((len(time_points), 6))
        for i, time in enumerate(time_points):
            earth_states[i] = - splined_moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(time)
        
        p.plot(xdata=earth_states[:, 0], ydata=earth_states[:, 1], color='black', alpha=0.3, linestyle='--')
        q.plot(xdata=earth_states[:, 0], ydata=earth_states[:, 2], color='black', alpha=0.3, linestyle='--')
    
    return p, q


# generate periodic orbits around unstable libration points
def periodic_orbit_continuation(periodic_orbit: pydylan.periodic_orbit, desired_value: float) -> pydylan.periodic_orbit:

    continuation_settings = pydylan.periodic_orbit.ContinuationParameterInfo()
    continuation_settings.parameter_t = pydylan.enum.ParameterType.energy
    continuation_settings.desired_value = desired_value
    continuation_settings.step_size = 1e-4
    continuation_settings.min_step_size = 1e-10
    periodic_orbit.set_continuation_information(continuation_settings, iLimit=1000)
    result = periodic_orbit.solve_for_orbit()

    return periodic_orbit


# generate a boundary condition using a state on EM L2 Lyapunov manifold arc
def generate_CR3BP_boundary_condition():

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

    pos_err, vel_err = 1E3, 1E-2

    libration_point_information_L2 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
    desired_orbit_energy_L2 = libration_point_information_L2[1] + 100E-4
    lyapunov_L2 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy_L2)
    lyapunov_L2 = periodic_orbit_continuation(lyapunov_L2, desired_orbit_energy_L2)

    L2_manifold_arc = lyapunov_L2.generate_manifold_arc(lyapunov_L2.orbit_period, 4., pydylan.enum.PerturbationDirection.StableRight)

    return pydylan.CR3BPStateBoundaryCondition(earth, moon, L2_manifold_arc.get_end_state(), pos_err, vel_err)


# convert rv state relative to moon to rv state relative to earth
def get_rv_state_relative_to_earth(epoch: float, states: np.array) -> (float, np.array):

    moon = pydylan.Body("Moon")

    if states.ndim==2:
        relative_state = np.copy(states)
        for i, ti in enumerate(epoch):
            relative_state[i, :6] = states[i, :6] + moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch[i])
        return epoch, relative_state
    else:
        return epoch, states + moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch)


# convert rv state relative to earth to rv state relative to moon
def get_rv_state_relative_to_moon(epoch: float, states: np.array) -> (float, np.array):

    moon = pydylan.Body("Moon")

    if states.ndim==2:
        relative_state = np.copy(states)
        for i, ti in enumerate(epoch):
            relative_state[i, :6] = states[i, :6] - moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch[i])
        return epoch, relative_state
    else:
        return epoch, states - moon.get_state_relative_to_parent_in_J2000_at_MJS_using_SPICE(epoch)


# convert rv state to coe state
def get_coe_state_from_rv_state(body: pydylan.Body, states: np.array) -> np.array:

    s2bp = pydylan.eom.S2BP(body)

    if states.ndim==2:
        coe_states = np.copy(states)
        for i, si in enumerate(states):
            tmp = s2bp.rv2coe(si[:3], si[3:6])
            coe_states[i, :6] = np.array([tmp.sma, tmp.ecc, tmp.inc, tmp.ran, tmp.ape, tmp.tru])
        return coe_states
    else:
        coe_states = s2bp.rv2coe(states[:3], states[3:6])
        return np.array([coe_states.sma, coe_states.ecc, coe_states.inc, coe_states.ran, coe_states.ape, coe_states.tru])


# convert coe state to rv state
def get_rv_state_from_coe_state(body: pydylan.Body, state: np.array) -> np.array:

    s2bp = pydylan.eom.S2BP(body)

    if state.ndim==2:
        coe_states = np.copy(state)
        for i, si in enumerate(state):
            sma, ecc, inc, lan, ape, tru = si[0], si[1], si[2], si[3], si[4], si[5]
            pos, vel = s2bp.coe2rv(sma, ecc, inc, lan, ape, tru)
            coe_states[i, :6] = np.concatenate((pos,vel))
        return coe_states
    else:
        sma, ecc, inc, lan, ape, tru = state[0], state[1], state[2], state[3], state[4], state[5]
        pos, vel = s2bp.coe2rv(sma, ecc, inc, lan, ape, tru)
        return np.concatenate((pos, vel))


# compute the relative error between two coe states
def compute_error_in_state(current_state: np.array, target_state: np.array, w: np.array):

    w = w / np.linalg.norm(w)
    err = w * (current_state - target_state)
    err = np.dot(err, err) / np.dot(target_state, target_state)

    return np.linalg.norm(err)


def read_qlaw_output(qlawhistory, thruster_parameters):

    qlawhistory = pd.read_csv(qlawhistory, names=["epoch", "sma", "ecc", "inc", "ape", "lan", "mass", "tru", "u0", "u1", "u2"])
    epoch, sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2 = qlawhistory.epoch, qlawhistory.sma, qlawhistory.ecc, qlawhistory.inc, qlawhistory.ape, qlawhistory.lan, qlawhistory.tru, qlawhistory.mass, qlawhistory.u0, qlawhistory.u1, qlawhistory.u2

    epoch, sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2 = epoch.to_numpy(), sma.to_numpy(), ecc.to_numpy(), inc.to_numpy(), ape.to_numpy(), lan.to_numpy(), tru.to_numpy() % (2 * np.pi), mass.to_numpy() - thruster_parameters.dry_mass, u0.to_numpy(), u1.to_numpy(), u2.to_numpy() 
    
    states = np.vstack((sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2)).T
    states = get_rv_state_from_coe_state(pydylan.Body("Moon"), states)

    return epoch, states


# obtain the trajectory from the NRHO_ORBIT to a LOW_LUNAR_ORBIT using a qlaw method
def solve_qlaw(initial_epoch, initial_state, thruster_parameters):

    print("======================================================", flush=True)
    print("PHASE III: Entering Lunar Science Orbit", flush=True)
    print("======================================================", flush=True)

    qlawhistory = "./qlaw_history.csv"
    if os.path.exists(qlawhistory): os.remove(qlawhistory)

    # used later for the total allowable integration time for qlaw
    t1, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    t2, _ = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()

    moon = pydylan.Body("Moon")
    initial_state = get_coe_state_from_rv_state(moon, initial_state)
    
    moon = pydylan.qlaw.body(moon.mu, moon.radius)
    coe_i = pydylan.qlaw.coe(initial_state[0], initial_state[1], initial_state[2] * RAD2DEG, initial_state[3] * RAD2DEG, initial_state[4] * RAD2DEG, initial_state[5] * RAD2DEG)
    coe_f = pydylan.qlaw.coe(4287, initial_state[1], 90)

    sc = pydylan.qlaw.thruster(thruster_parameters.thrust, thruster_parameters.Isp, thruster_parameters.fuel_mass + thruster_parameters.dry_mass)

    settings = pydylan.qlaw.QLaw_Parameters()
    
    settings.tol_sma = 10.
    settings.tol_inc = 1
    settings.tol_ecc = 0.1
    settings.Wsma = 0.45
    settings.Wecc = 0.1
    settings.Winc = 0.45
    settings.Wape = 0.0
    settings.Wlan = 0.0
    settings.Wp = 0.0
    settings.tld_limit = 2 * (t2 - t1)
    settings.stepsize = 1E-3
    settings.minimum_periapse = pydylan.Body("Moon").radius * 2
    settings.t0 = initial_epoch

    qlawsolver = pydylan.qlaw.QLaw5_Solver(moon, coe_i, coe_f, sc, settings)
    qlawsolver.get_qlaw_history = True
    qlawsolver.history_filename = os.getcwd()+"/qlaw_history.csv"

    # print(f"INITIAL COE -> sma: {initial_state[0]} km, ecc: {initial_state[1]}, inc: {initial_state[2] * RAD2DEG}, arg_per: {initial_state[3] * RAD2DEG}, lon_asc: {initial_state[4] * RAD2DEG}, tru_ano: {(initial_state[5] * RAD2DEG) % 360}")

    final_state = qlawsolver.solve()

    sma = final_state[0] / (1 - final_state[1] * final_state[1]) * moon.DU
    ecc = final_state[1]
    inc = final_state[2] * 180 / np.pi

    # print(f"FINAL COE -> sma: {sma} km, ecc: {ecc}, inc: {final_state[2] * RAD2DEG}, arg_per: {final_state[3] * RAD2DEG}, lon_asc: {final_state[4] * RAD2DEG}, tru_ano: {(final_state[6] * RAD2DEG) % 360}")

    assert(qlawsolver.status == pydylan.qlaw.QLaw5_Solver_Status.converged)
    assert(abs(sma - coe_f.sma) < settings.tol_sma)
    assert(abs(ecc - coe_f.ecc) < settings.tol_ecc)
    assert(abs(inc - coe_f.inc) < settings.tol_inc)

    epoch, states = read_qlaw_output(qlawhistory, thruster_parameters)

    return epoch, states


# obtain the trajectory from the NRHO_INSERTION_POINT to a NRHO_ORBIT by thrusting in the velocity direction
def solve_spiral(eom: pydylan.eom, thruster_parameters:pydylan.ThrustParameters, initial_epoch, initial_state, verbose=False) -> (np.ndarray, np.ndarray):

    print("======================================================", flush=True)
    print("PHASE II: Entering NRHO", flush=True)
    print("======================================================", flush=True)

    moon = pydylan.Body("Moon")

    initial_epoch, initial_state = get_rv_state_relative_to_moon(initial_epoch, initial_state)
    current_epoch, current_state = initial_epoch, initial_state

    error_weight = np.array([0.45, 0.1, 0.45, 0., 0., 0.])
    
    target_coe_state = np.array([4287 * 2.0, 0.5714, np.pi / 2, 0., 0., 0.])
    current_coe_state = get_coe_state_from_rv_state(moon, current_state)
    current_error = compute_error_in_state(current_coe_state, target_coe_state, error_weight)

    # First, compute the time-of-flight to reach within a tolerance of the desired coe state
    time_of_flight, dt, niter = 0, 100, 0

    time, states = np.zeros((dt + 1,)), np.zeros((dt + 1, 10))
    while current_error > 5E-2 and niter<1E6:

        if ((niter % 100) == 0 and verbose): print(f"Iteration: {niter}, current error: {current_error}")

        outward_spiral = True if (target_coe_state[0] > current_coe_state[0]) else False

        time_of_flight = current_epoch + dt

        spiral_onto_NRHO = pydylan.phases.lowthrust_spiral(eom, current_state, thruster_parameters)
        spiral_onto_NRHO.evaluate(time_of_flight, outward_spiral, current_epoch, dt * 1E-1)
        
        current_epoch, current_state = time_of_flight, spiral_onto_NRHO.get_final_states(False)
        current_coe_state = get_coe_state_from_rv_state(moon, current_state)
        current_error = compute_error_in_state(current_coe_state, target_coe_state, error_weight)

        time = np.concatenate((time, spiral_onto_NRHO.get_time()))
        states = np.concatenate((states, spiral_onto_NRHO.get_states()), axis=0)

        niter += 1

    time, states = time[dt + 1:], states[dt + 1:, :]

    return np.array(time), np.array(states)


# obtain the trajectory from the POST_DEPLOYMENT_STATE to NRHO_INSERTION_STATE with Fixed+Fixed+Fixed BCs
def solve_transfer_fixed_BC(eom: pydylan.eom, thruster_parameters: pydylan.ThrustParameters) -> (np.ndarray, np.ndarray):

    print("======================================================", flush=True)
    print("PHASE I: Solving the waypoint tracking phase with Fixed-Fixed-Fixed BC", flush=True)
    print("======================================================", flush=True)

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.FixedBoundaryCondition(middle_state)
    right_boundary_condition = pydylan.FixedBoundaryCondition(target_state)

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = True
    snopt_options.time_limit = 1

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 1.
    mbh_options.quiet_MBH = True
    mbh_options.time_limit = 1

    phase_options_1 = pydylan.phase_options_structure()
    phase_options_1.number_of_segments = 10

    phase_options_1.earliest_initial_date_in_MJS = initial_epoch
    phase_options_1.minimum_initial_coast_time = 0.
    phase_options_1.maximum_initial_coast_time = 0.
    phase_options_1.minimum_final_coast_time = 0.
    phase_options_1.maximum_final_coast_time = 0.
    phase_options_1.minimum_shooting_time = middle_epoch - initial_epoch
    phase_options_1.maximum_shooting_time = middle_epoch - initial_epoch

    phase_options_1.match_point_position_constraint_tolerance = 1E-12
    phase_options_1.match_point_velocity_constraint_tolerance = 1E-12
    phase_options_1.match_point_mass_constraint_tolerance = 1E-9
    phase_options_1.control_coordinate_transcription = pydylan.enum.polar

    phase_options_2 = pydylan.phase_options_structure()
    phase_options_2.number_of_segments = 100
    
    phase_options_2.earliest_initial_date_in_MJS = middle_epoch
    phase_options_2.minimum_initial_coast_time = 0.
    phase_options_2.maximum_initial_coast_time = 0.
    phase_options_2.minimum_final_coast_time = 0.
    phase_options_2.maximum_final_coast_time = 0.
    phase_options_2.minimum_shooting_time = target_epoch - middle_epoch
    phase_options_2.maximum_shooting_time = target_epoch - middle_epoch

    phase_options_2.match_point_position_constraint_tolerance = 1E-12
    phase_options_2.match_point_velocity_constraint_tolerance = 1E-12
    phase_options_2.match_point_mass_constraint_tolerance = 1E-9
    phase_options_2.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, middle_boundary_condition, pydylan.enum.mbh)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_phase_options(phase_options_1)
    mission.add_control_initial_guess(p1_initial_guess_fixed_BC())
    mission.add_boundary_condition(right_boundary_condition)
    mission.add_phase_options(phase_options_2)
    mission.add_control_initial_guess(p2_initial_guess_fixed_BC())
    mission.optimize(snopt_options, mbh_options)

    # output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states


# obtain the trajectory from the POST_DEPLOYMENT_STATE to NRHO_INSERTION_STATE with Fixed+Advecting+CR3BP BCs
def solve_transfer_other_BC(eom, thruster_parameters) -> (np.ndarray, np.ndarray):

    print("======================================================", flush=True)
    print("PHASE I: Solving the waypoint tracking phase with Fixed-Advecting-CR3BP BC", flush=True)
    print("======================================================", flush=True)

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.AdvectingBoundaryCondition(middle_state, middle_epoch, eom)
    right_boundary_condition = generate_CR3BP_boundary_condition()

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = True
    snopt_options.time_limit = 1

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    # mbh_options.max_step_size = 1.
    mbh_options.quiet_MBH = True
    mbh_options.time_limit = 1

    phase_options_1 = pydylan.phase_options_structure()
    phase_options_1.number_of_segments = 10

    phase_options_1.earliest_initial_date_in_MJS = initial_epoch
    phase_options_1.minimum_initial_coast_time = 0.
    phase_options_1.maximum_initial_coast_time = 0.
    phase_options_1.minimum_final_coast_time = 0.
    phase_options_1.maximum_final_coast_time = 0.
    phase_options_1.minimum_shooting_time = 2 * (middle_epoch - initial_epoch) / 3
    phase_options_1.maximum_shooting_time = 4 * (middle_epoch - initial_epoch) / 3

    phase_options_1.match_point_position_constraint_tolerance = 1E-12
    phase_options_1.match_point_velocity_constraint_tolerance = 1E-12
    phase_options_1.match_point_mass_constraint_tolerance = 1E-9
    phase_options_1.control_coordinate_transcription = pydylan.enum.polar

    phase_options_2 = pydylan.phase_options_structure()
    phase_options_2.number_of_segments = 100
    
    phase_options_2.earliest_initial_date_in_MJS = middle_epoch
    phase_options_2.minimum_initial_coast_time = 0.
    phase_options_2.maximum_initial_coast_time = 0.
    phase_options_2.minimum_final_coast_time = 0.
    phase_options_2.maximum_final_coast_time = 0.
    phase_options_2.minimum_shooting_time = target_epoch - middle_epoch - (middle_epoch - initial_epoch) / 3
    phase_options_2.maximum_shooting_time = target_epoch - middle_epoch + (middle_epoch - initial_epoch) / 3

    phase_options_2.match_point_position_constraint_tolerance = 1E-6
    phase_options_2.match_point_velocity_constraint_tolerance = 1E-6
    phase_options_2.match_point_mass_constraint_tolerance = 1E-3
    phase_options_2.control_coordinate_transcription = pydylan.enum.polar

    mission = pydylan.Mission(eom, left_boundary_condition, middle_boundary_condition, pydylan.enum.mbh)
    mission.set_thruster_parameters(thruster_parameters)
    mission.add_phase_options(phase_options_1)
    mission.add_control_initial_guess(p1_initial_guess_other_BC())
    mission.add_boundary_condition(right_boundary_condition)
    mission.add_phase_options(phase_options_2)
    mission.add_control_initial_guess(p2_initial_guess_other_BC())
    mission.optimize(snopt_options, mbh_options)

    # output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states


def setup_dynamical_system():

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    sun = pydylan.Body("Sun")

    eom_earth = pydylan.eom.Ephemeris_nBP(earth)
    eom_earth.add_secondary_body(moon)
    eom_earth.add_secondary_body(sun)
    
    eom_moon = pydylan.eom.Ephemeris_nBP(moon)
    eom_moon.add_secondary_body(earth)
    eom_moon.add_secondary_body(sun)

    return earth, moon, eom_earth, eom_moon


def solve_phase(phase, boundary_condition):

    earth, _, eom_earth, eom_moon = setup_dynamical_system()

    thruster_parameters = pydylan.ThrustParameters(fuel_mass=1.5, dry_mass=12.5, Isp=2500., thrust=1.5E-3)

    if boundary_condition.lower()=='fixed':
        time_transfer, states_transfer_rel_earth = solve_transfer_fixed_BC(eom_earth, thruster_parameters)
    else:
        time_transfer, states_transfer_rel_earth = solve_transfer_other_BC(eom_earth, thruster_parameters)
    _, states_transfer_rel_moon = get_rv_state_relative_to_moon(time_transfer, states_transfer_rel_earth)

    result_dict = {
            'time_transfer': time_transfer,
            'states_transfer_rel_earth': states_transfer_rel_earth,
            'states_transfer_rel_moon': states_transfer_rel_moon}

    result_filepath = "/Users/amlansinha/Workspace/Princeton/Beeson/Library/VISUALIZATION_TOOLS/LIC/kernels" + "/LIC_rel_to_earth.bsp"
    if os.path.exists(result_filepath): os.remove(result_filepath)
    file_writer = pydylan.spice.SpiceFileWriter(result_filepath, False)
    unique_time_transfer, unique_transfer_idx = np.unique(time_transfer, return_index=True)
    file_writer.write_segment(unique_time_transfer, states_transfer_rel_earth[unique_transfer_idx, :6], -12345, earth.spiceId, "PHASEI")
    file_writer.close()

    if phase==1:
        thruster_parameters.fuel_mass = states_transfer_rel_earth[-1,6]
        time_spiral, states_spiral_rel_moon = solve_spiral(eom_moon, thruster_parameters, time_transfer[-1], states_transfer_rel_earth[-1, :6])
        _, states_spiral_rel_earth = get_rv_state_relative_to_earth(time_spiral, states_spiral_rel_moon)

        result_dict['time_spiral'] = time_spiral
        result_dict['states_spiral_rel_earth'] = states_spiral_rel_earth
        result_dict['states_spiral_rel_moon'] = states_spiral_rel_moon

        file_writer = pydylan.spice.SpiceFileWriter(result_filepath, True)
        unique_time_spiral, unique_spiral_idx = np.unique(time_spiral, return_index=True)
        file_writer.write_segment(unique_time_spiral, states_spiral_rel_earth[unique_spiral_idx, :6], -12345, earth.spiceId, "PHASEII")
        file_writer.close()

    elif phase>=2:

        thruster_parameters.fuel_mass = states_transfer_rel_earth[-1,6]
        time_spiral, states_spiral_rel_moon = solve_spiral(eom_moon, thruster_parameters, time_transfer[-1], states_transfer_rel_earth[-1, :6])

        thruster_parameters.fuel_mass = states_spiral_rel_moon[-1,6]
        time_qlaw, states_qlaw_rel_moon = solve_qlaw(time_spiral[-1], states_spiral_rel_moon[-1, :6], thruster_parameters)

        _, states_spiral_rel_earth = get_rv_state_relative_to_earth(time_spiral, states_spiral_rel_moon)
        _, states_qlaw_rel_earth = get_rv_state_relative_to_earth(time_qlaw, states_qlaw_rel_moon)

        result_dict['time_spiral'] = time_spiral
        result_dict['states_spiral_rel_earth'] = states_spiral_rel_earth
        result_dict['states_spiral_rel_moon'] = states_spiral_rel_moon

        result_dict['time_qlaw'] = time_qlaw
        result_dict['states_qlaw_rel_earth'] = states_qlaw_rel_earth
        result_dict['states_qlaw_rel_moon'] = states_qlaw_rel_moon

        file_writer = pydylan.spice.SpiceFileWriter(result_filepath, True)
        unique_time_spiral, unique_spiral_idx = np.unique(time_spiral, return_index=True)
        file_writer.write_segment(unique_time_spiral, states_spiral_rel_earth[unique_spiral_idx, :6], -12345, earth.spiceId, "PHASEII")
        unique_time_qlaw, unique_qlaw_idx = np.unique(time_qlaw, return_index=True)
        file_writer.write_segment(unique_time_qlaw, states_qlaw_rel_earth[unique_qlaw_idx, :6], -12345, earth.spiceId, "PHASEIII")
        file_writer.close()

    return result_dict


def plot_solutions(result_dict, phase, boundary_condition, verbose=True):

    earth, moon, eom_earth, eom_moon = setup_dynamical_system()

    p, q = get_plots_of_phase_integration(earth, eom_earth)
    r, s = get_plots_of_phase_integration(moon, eom_moon)

    initial_epoch, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()

    d1, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    d1 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d1)
    d2, _ = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    d2 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d2)
    d3, _ = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()
    d3 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d3)

    print("======================================================", flush=True)
    print("MISSION PARAMETERS", flush=True)
    print("======================================================", flush=True)

    if phase==0:

        time_transfer, states_transfer_rel_earth, states_transfer_rel_moon = result_dict.get('time_transfer'), result_dict.get('states_transfer_rel_earth'), result_dict.get('states_transfer_rel_moon')

        p.plot(xdata=states_transfer_rel_earth[:, 0], ydata=states_transfer_rel_earth[:, 1], color='black', linewidth=1)
        q.plot(xdata=states_transfer_rel_earth[:, 0], ydata=states_transfer_rel_earth[:, 2], color='black', linewidth=1)

        r.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 1], color='black', linewidth=1)
        s.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 2], color='black', linewidth=1)

        time_transfer = (time_transfer - initial_epoch) * SEC2DAY

        t = LinePlot()
        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_transfer, ydata=get_throttle_history(states_transfer_rel_earth[:, -3:]), color='black', linewidth=2)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_transfer, ydata=states_transfer_rel_earth[:, 6], color='black', linewidth=2)

        print('\n Transfer time: {} (days)'.format(time_transfer[-1]))
        print('\n DEPLOYMENT: ', d1.year, '-', d1.month, '-', d1.day, '|', d1.hour, ':', d1.minute, ':', d1.second)
        print('\n POST-FLYBY: ', d2.year, '-', d2.month, '-', d2.day, '|', d2.hour, ':', d2.minute, ':', d2.second)
        print('\n NRHO-INSERTION: ', d3.year, '-', d3.month, '-', d3.day, '|', d3.hour, ':', d3.minute, ':', d3.second)

        print('\n Mass Remaining', states_transfer_rel_earth[-1, 6], ' kg')

    elif phase==1:

        time_transfer, states_transfer_rel_earth, states_transfer_rel_moon = result_dict.get('time_transfer'), result_dict.get('states_transfer_rel_earth'), result_dict.get('states_transfer_rel_moon')
        time_spiral, states_spiral_rel_earth, states_spiral_rel_moon = result_dict.get('time_spiral'), result_dict.get('states_spiral_rel_earth'), result_dict.get('states_spiral_rel_moon')

        p.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 1], color=strongorange, linewidth=1, alpha=0.9)
        q.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 2], color=strongorange, linewidth=1, alpha=0.9)

        r.plot(xdata=states_spiral_rel_moon[:, 0], ydata=states_spiral_rel_moon[:, 1], color=strongorange, linewidth=1, alpha=0.9)
        s.plot(xdata=states_spiral_rel_moon[:, 0], ydata=states_spiral_rel_moon[:, 2], color=strongorange, linewidth=1, alpha=0.9)

        if boundary_condition.lower()=='fixed':
            r.set_axis('tight', [-12000, 12000, -20000, 50000])
            s.set_axis('tight', [-50000, 50000, -60000, 25000])
        else:
            r.set_axis('tight', [-30000, 60000, -40000, 40000])
            s.set_axis('tight', [-30000, 60000, -15000, 20000])

        d4 = time_spiral[-1]
        d4 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d4)

        print('\n Spiral time: {} (days)'.format(time_spiral[-1] - time_transfer[-1]))
        print('\n DEPLOYMENT: ', d1.year, '-', d1.month, '-', d1.day, '|', d1.hour, ':', d1.minute, ':', d1.second)
        print('\n POST-FLYBY: ', d2.year, '-', d2.month, '-', d2.day, '|', d2.hour, ':', d2.minute, ':', d2.second)
        print('\n NRHO-INSERTION: ', d3.year, '-', d3.month, '-', d3.day, '|', d3.hour, ':', d3.minute, ':', d3.second)
        print('\n END OF SPIRAL: ', d4.year, '-', d4.month, '-', d4.day, '|', d4.hour, ':', d4.minute, ':', d4.second)

        print('\n Mass Remaining', states_spiral_rel_earth[-1, 6], ' kg')

    elif phase==2:
        
        time_transfer, states_transfer_rel_earth, states_transfer_rel_moon = result_dict.get('time_transfer'), result_dict.get('states_transfer_rel_earth'), result_dict.get('states_transfer_rel_moon')
        time_spiral, states_spiral_rel_earth, states_spiral_rel_moon = result_dict.get('time_spiral'), result_dict.get('states_spiral_rel_earth'), result_dict.get('states_spiral_rel_moon')
        time_qlaw, states_qlaw_rel_earth, states_qlaw_rel_moon = result_dict.get('time_qlaw'), result_dict.get('states_qlaw_rel_earth'), result_dict.get('states_qlaw_rel_moon')

        p.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 1], color=babyblue, linewidth=1, alpha=0.8)
        q.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 2], color=babyblue, linewidth=1, alpha=0.8)

        r.plot(xdata=states_qlaw_rel_moon[:, 0], ydata=states_qlaw_rel_moon[:, 1], color=babyblue, linewidth=1, alpha=0.7)
        s.plot(xdata=states_qlaw_rel_moon[:, 0], ydata=states_qlaw_rel_moon[:, 2], color=babyblue, linewidth=1, alpha=0.7)

        if boundary_condition.lower()=='fixed':
            r.set_axis('tight', [-12000, 12000, -20000, 50000])
            s.set_axis('tight', [-50000, 50000, -60000, 25000])
        else:
            r.set_axis('tight', [-30000, 60000, -40000, 40000])
            s.set_axis('tight', [-30000, 60000, -15000, 20000])

        d4 = time_spiral[-1]
        d4 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d4)
        d5 = time_qlaw[-1]
        d5 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d5)

        print('\n QLaw time: {} (days)'.format(time_qlaw[-1] - time_spiral[-1]))
        print('\n Spiral time: {} (days)'.format(time_spiral[-1] - time_transfer[-1]))
        print('\n DEPLOYMENT: ', d1.year, '-', d1.month, '-', d1.day, '|', d1.hour, ':', d1.minute, ':', d1.second)
        print('\n POST-FLYBY: ', d2.year, '-', d2.month, '-', d2.day, '|', d2.hour, ':', d2.minute, ':', d2.second)
        print('\n NRHO-INSERTION: ', d3.year, '-', d3.month, '-', d3.day, '|', d3.hour, ':', d3.minute, ':', d3.second)
        print('\n END OF SPIRAL: ', d4.year, '-', d4.month, '-', d4.day, '|', d4.hour, ':', d4.minute, ':', d4.second)
        print('\n END OF QLAW: ', d5.year, '-', d5.month, '-', d5.day, '|', d5.hour, ':', d5.minute, ':', d5.second)

        print('\n Mass Remaining', states_qlaw_rel_earth[-1, 6], ' kg')

    else:

        time_transfer, states_transfer_rel_earth, states_transfer_rel_moon = result_dict.get('time_transfer'), result_dict.get('states_transfer_rel_earth'), result_dict.get('states_transfer_rel_moon')
        time_spiral, states_spiral_rel_earth, states_spiral_rel_moon = result_dict.get('time_spiral'), result_dict.get('states_spiral_rel_earth'), result_dict.get('states_spiral_rel_moon')
        time_qlaw, states_qlaw_rel_earth, states_qlaw_rel_moon = result_dict.get('time_qlaw'), result_dict.get('states_qlaw_rel_earth'), result_dict.get('states_qlaw_rel_moon')

        p.plot(xdata=states_transfer_rel_earth[:, 0], ydata=states_transfer_rel_earth[:, 1], color='black', linewidth=1)
        q.plot(xdata=states_transfer_rel_earth[:, 0], ydata=states_transfer_rel_earth[:, 2], color='black', linewidth=1)
        p.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 1], color=strongorange, linewidth=1, alpha=0.9)
        q.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 2], color=strongorange, linewidth=1, alpha=0.9)
        p.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 1], color=babyblue, linewidth=1, alpha=0.8)
        q.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 2], color=babyblue, linewidth=1, alpha=0.8)

        r.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 1], color='black', linewidth=1)
        s.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 2], color='black', linewidth=1)
        r.plot(xdata=states_spiral_rel_moon[:, 0], ydata=states_spiral_rel_moon[:, 1], color=strongorange, linewidth=1, alpha=0.9)
        s.plot(xdata=states_spiral_rel_moon[:, 0], ydata=states_spiral_rel_moon[:, 2], color=strongorange, linewidth=1, alpha=0.9)
        r.plot(xdata=states_qlaw_rel_moon[:, 0], ydata=states_qlaw_rel_moon[:, 1], color=babyblue, linewidth=1, alpha=0.7)
        s.plot(xdata=states_qlaw_rel_moon[:, 0], ydata=states_qlaw_rel_moon[:, 2], color=babyblue, linewidth=1, alpha=0.7)

        if boundary_condition.lower()=='fixed':
            r.set_axis('tight', [-12000, 12000, -20000, 50000])
            s.set_axis('tight', [-50000, 50000, -60000, 25000])
        else:
            r.set_axis('tight', [-30000, 60000, -40000, 40000])
            s.set_axis('tight', [-30000, 60000, -15000, 20000])

        d4 = time_spiral[-1]
        d4 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d4)
        d5 = time_qlaw[-1]
        d5 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d5)

        time_transfer, time_spiral, time_qlaw = (time_transfer - initial_epoch) * SEC2DAY, (time_spiral - initial_epoch) * SEC2DAY, (time_qlaw - initial_epoch) * SEC2DAY

        t = LinePlot()
        t.grid()
        t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
        t.plot(xdata=time_transfer, ydata=get_throttle_history(states_transfer_rel_earth[:, -3:]), color='black', linewidth=2)

        m = LinePlot()
        m.grid()
        m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
        m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
        m.plot(xdata=time_transfer, ydata=states_transfer_rel_earth[:, 6], color='black', linewidth=2)

        fig, axs = plt.subplots(3, 2, sharex=True)
        axs[0, 0].plot(time_transfer, states_transfer_rel_earth[:, 0])
        axs[0, 0].plot(time_spiral, states_spiral_rel_earth[:, 0], color=strongorange)
        axs[0, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 0], color=babyblue)
        axs[0, 0].set_title('X-coordinate')
        axs[0, 0].grid()
        axs[1, 0].plot(time_transfer, states_transfer_rel_earth[:, 1])
        axs[1, 0].plot(time_spiral, states_spiral_rel_earth[:, 1], color=strongorange)
        axs[1, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 1], color=babyblue)
        axs[1, 0].set_title('Y-coordinate')
        axs[1, 0].grid()
        axs[2, 0].plot(time_transfer, states_transfer_rel_earth[:, 2])
        axs[2, 0].plot(time_spiral, states_spiral_rel_earth[:, 2], color=strongorange)
        axs[2, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 2], color=babyblue)
        axs[2, 0].set_title('Z-coordinate')
        axs[2, 0].grid()
        axs[0, 1].plot(time_transfer, states_transfer_rel_earth[:, 3])
        axs[0, 1].plot(time_spiral, states_spiral_rel_earth[:, 3], color=strongorange)
        axs[0, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 3], color=babyblue)
        axs[0, 1].set_title('VX-coordinate')
        axs[0, 1].grid()
        axs[1, 1].plot(time_transfer, states_transfer_rel_earth[:, 4])
        axs[1, 1].plot(time_spiral, states_spiral_rel_earth[:, 4], color=strongorange)
        axs[1, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 4], color=babyblue)
        axs[1, 1].set_title('VY-coordinate')
        axs[1, 1].grid()
        axs[2, 1].plot(time_transfer, states_transfer_rel_earth[:, 5])
        axs[2, 1].plot(time_spiral, states_spiral_rel_earth[:, 5], color=strongorange)
        axs[2, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 5], color=babyblue)
        axs[2, 1].set_title('VZ-coordinate')
        axs[2, 1].grid()

        print('\n Transfer time: {} (days)'.format(time_transfer[-1]))
        print('\n Spiral time: {} (days)'.format(time_spiral[-1] - time_transfer[-1]))
        print('\n QLaw time: {} (days)'.format(time_qlaw[-1] - time_spiral[-1]))
        
        print('\n DEPLOYMENT: ', d1.year, '-', d1.month, '-', d1.day, '|', d1.hour, ':', d1.minute, ':', d1.second)
        print('\n POST-FLYBY: ', d2.year, '-', d2.month, '-', d2.day, '|', d2.hour, ':', d2.minute, ':', d2.second)
        print('\n NRHO-INSERTION: ', d3.year, '-', d3.month, '-', d3.day, '|', d3.hour, ':', d3.minute, ':', d3.second)

        print('\n END OF SPIRAL: ', d4.year, '-', d4.month, '-', d4.day, '|', d4.hour, ':', d4.minute, ':', d4.second)
        print('\n END OF QLAW: ', d5.year, '-', d5.month, '-', d5.day, '|', d5.hour, ':', d5.minute, ':', d5.second)

        print('\n Mass Remaining', states_qlaw_rel_earth[-1, 6], ' kg')

    plt.show()

if __name__ == '__main__':

    # Load SPICE kernels
    kernels = pydylan.spice.load_spice()

    phase = int(input("Which phase do you want to solve for? Enter an integer between 0 and 2. (0 for Phase I (until NRHO), 1 for Phase II (entering NRHO), 2 for Phase III (entering LLO) and >2 for the full mission) \n"))

    boundary_condition = input("Which boundary condition do you want to impose at the end of Phase I? Enter 'fixed' or 'other' \n")

    results = solve_phase(phase, boundary_condition)
    plot_solutions(results, phase, boundary_condition, verbose=True)

    # Unload SPICE kernels
    pydylan.spice.unload_spice(kernels)
