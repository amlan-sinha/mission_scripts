import sys, os

sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding/debug')
sys.path.append('/Users/amlansinha/Workspace/Princeton/Beeson/DyLAN/dylan/build/python_binding')

import pydylan
import numpy as np
import pandas as pd
from GSFC_reference_waypoints import get_post_deployment_epoch_in_MJS_and_state_in_EJ2000, get_post_flyby_epoch_in_MJS_and_state_in_EJ2000, get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000

import matplotlib.pyplot as plt
from lineplot import LinePlot
from support import html_colors

DAY2SEC, SEC2DAY, RAD2DEG = 86400., 1/86400., 180 / np.pi

strongorange, babyblue = html_colors['strong orange'], html_colors['baby blue']


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase I)
def p1_initial_guess_fixed_BC():

    return np.array([482482.8659490347,
                        0.0,
                        0.0,
                        1.7464184341069648,
                        0.0,
                        0.4331765163514802,
                        1.7473093800000001,
                        0.0,
                        1.0,
                        1.7536222300000002,
                        0.0,
                        0.743695465,
                        1.8552170800000003,
                        0.0,
                        0.0,
                        6.283185307179586,
                        4.52776291,
                        0.06023392968055219,
                        5.76438479,
                        4.56470582,
                        0.0,
                        4.929825823569898,
                        2.7223116318768477,
                        0.519208856261034,
                        6.283185307179586,
                        6.283185307179586,
                        0.0,
                        6.283185307179586,
                        5.90283121,
                        0.0,
                        0.0,
                        4.62921218,
                        0.0,
                        1.4928636047043378], dtype=np.longdouble)


# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase I)
# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase I)
def p1_initial_guess_other_BC():

    return np.array([450249.3938213671,
                    0.0,
                    0.0,
                    1.0172384497371907,
                    5.580414883585732,
                    0.5583187679869422,
                    4.5556611164131855,
                    2.9736498043610893,
                    1.0,
                    4.852232615388868,
                    2.955753129459565,
                    1.0,
                    0.0003465947676473578,
                    3.0677963582167127,
                    0.9976537546258777,
                    6.183582327869987,
                    0.0,
                    0.0005419654923028792,
                    0.0,
                    6.283185307179586,
                    0.007021913616382374,
                    5.415748347524338,
                    3.036391756493991,
                    0.9936234178612346,
                    0.009163220290764947,
                    3.1202968609213237,
                    0.9977836109863358,
                    0.0,
                    0.0,
                    0.002101052143183159,
                    0.0,
                    0.0,
                    0.0022131805628988256,
                    1.4846861269118294], dtype=np.longdouble)


# Feasible initial condition with Fixed+Fixed+Fixed BC (Phase II)
def p2_initial_guess_fixed_BC():

    return np.array([15532164.364881873,
                        0.0,
                        0.0,
                        5.0671042,
                        5.44995698,
                        0.0,
                        6.283185307179586,
                        3.21637982,
                        0.02913733722539787,
                        3.9760579499999995,
                        0.0,
                        0.0,
                        3.79683439,
                        5.4308003,
                        0.0,
                        0.0,
                        0.298567402,
                        0.0,
                        0.0,
                        5.30308886,
                        0.0,
                        0.0,
                        4.19636311,
                        0.0,
                        0.0,
                        3.7079902600000003,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2.70455048,
                        0.0,
                        0.0,
                        2.87081524,
                        1.43198238,
                        0.0,
                        3.08208809,
                        0.544863448,
                        0.0,
                        2.42291946,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        6.15220744,
                        3.06598264,
                        0.0,
                        6.21362082,
                        2.98845422,
                        0.0,
                        6.15834144,
                        2.95730207,
                        0.0,
                        2.95964976,
                        0.159074879,
                        1.0,
                        0.0319331999,
                        6.283185307179586,
                        0.0,
                        3.3309672600000004,
                        1.18052119,
                        0.0,
                        2.96994906,
                        0.17601555,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        2.97774731,
                        6.283185307179586,
                        1.0,
                        6.12330783,
                        2.95047661,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        3.99327095,
                        0.0,
                        0.0,
                        0.0,
                        2.93317499,
                        0.9798121234135689,
                        2.99625342,
                        0.211366675,
                        0.0,
                        0.0,
                        3.0202044,
                        0.0,
                        0.0,
                        2.80910262,
                        0.0,
                        2.91706763,
                        0.283680515,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2.9529144,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2.98168501,
                        0.237951659,
                        0.0,
                        0.0,
                        0.131180744,
                        0.0,
                        6.283185307179586,
                        3.37938523,
                        0.0,
                        2.93618851,
                        0.24743698599999997,
                        0.0,
                        0.0,
                        2.94076802,
                        0.0,
                        6.283185307179586,
                        6.283185307179586,
                        0.0,
                        0.43463698,
                        0.0,
                        0.0,
                        0.0,
                        4.27653937,
                        0.0,
                        6.283185307179586,
                        0.82754145,
                        0.0,
                        6.283185307179586,
                        4.20230017,
                        0.0,
                        5.13384386,
                        6.283185307179586,
                        0.0,
                        5.35562739,
                        0.0489496897,
                        0.0,
                        6.283185307179586,
                        0.0,
                        0.0,
                        0.0,
                        6.283185307179586,
                        0.0,
                        3.50066706,
                        5.72758229,
                        0.0,
                        0.0,
                        3.79420115,
                        0.0,
                        1.94775704,
                        5.12580325,
                        0.0,
                        2.01179308,
                        3.2167639899999996,
                        0.0,
                        0.0,
                        3.8321413500000006,
                        0.0412405121967852,
                        4.21785246,
                        1.08452055,
                        0.0,
                        0.0,
                        6.283185307179586,
                        0.0,
                        6.283185307179586,
                        6.283185307179586,
                        0.0,
                        1.39405117,
                        0.64785571,
                        0.0,
                        6.283185307179586,
                        6.283185307179586,
                        0.0,
                        3.14058623,
                        5.72287417,
                        0.0,
                        1.11862527,
                        0.383633765,
                        0.0,
                        0.0,
                        5.11099849,
                        0.0,
                        4.35129692,
                        2.86438994,
                        0.0,
                        6.283185307179586,
                        3.3831514699999996,
                        0.0,
                        3.5314500700000004,
                        2.21651767,
                        0.0,
                        4.16806746,
                        2.866360916544391,
                        1.0,
                        0.9740019480000001,
                        6.283185307179586,
                        1.0,
                        0.9262198940000002,
                        0.260804978,
                        1.0,
                        1.48343033,
                        0.716495579,
                        0.0,
                        0.0,
                        5.00614557,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.186661748,
                        0.0,
                        0.744227401,
                        0.23022753500000004,
                        1.0,
                        0.0,
                        4.15374384,
                        0.0,
                        0.0,
                        0.9449652050000001,
                        0.0,
                        0.667571104,
                        0.21854164277437113,
                        1.0,
                        6.283185307179586,
                        0.274210129,
                        0.0,
                        0.6261067517430549,
                        6.283185307179586,
                        1.0,
                        6.283185307179586,
                        6.283185307179586,
                        0.0,
                        6.283185307179586,
                        0.0,
                        0.0,
                        0.0,
                        2.20059806,
                        0.0,
                        3.70378762,
                        2.92640837,
                        1.0,
                        0.549828651,
                        0.21787073699999998,
                        0.368307212683778,
                        2.75793027,
                        6.283185307179586,
                        0.0,
                        0.0,
                        6.283185307179586,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.52768998,
                        0.0,
                        6.283185307179586,
                        5.61457043,
                        0.0,
                        6.283185307179586,
                        0.0,
                        0.0,
                        0.8694460070000001,
                        0.400714413,
                        0.0,
                        5.89215347,
                        0.104677045,
                        0.0,
                        0.264101977,
                        6.283185307179586,
                        0.0,
                        4.34901505,
                        2.73451232,
                        0.0,
                        0.571121871,
                        6.283185307179586,
                        0.0,
                        0.0,
                        6.283185307179586,
                        0.0,
                        0.111198684,
                        6.283185307179586,
                        0.0,
                        1.42220863,
                        1.33517987,
                        0.0,
                        2.47180317,
                        5.19099338,
                        0.0,
                        5.28589429,
                        2.88722504,
                        0.0,
                        1.68199427,
                        2.0116499,
                        0.0,
                        6.0226874,
                        3.13339445,
                        0.0,
                        1.375854231219523], dtype=np.longdouble)


# Feasible initial condition with Fixed+Advecting+CR3BP BC (Phase II)
def p2_initial_guess_other_BC():

    return np.array([15395395.867286364,
                    0.0,
                    0.0,
                    0.00884986108821831,
                    0.002481989964316016,
                    0.01312331918554677,
                    0.02087189926323989,
                    0.031883374053028546,
                    0.14455481236177312,
                    0.004474997086940683,
                    0.0030250651808949236,
                    0.009547426461270606,
                    2.1750109717799604,
                    4.136200651308372,
                    0.5796672610210561,
                    4.653796917656941,
                    0.07397570233865397,
                    0.7978706384415972,
                    0.004611892872999658,
                    0.004034726890334901,
                    0.016497400778396284,
                    2.6757177600304987,
                    0.0,
                    0.007852163712517356,
                    0.3598743223228568,
                    0.0,
                    0.012574466201907835,
                    0.0,
                    3.537176981322024,
                    0.010202087620546063,
                    0.08043594829987759,
                    0.0939021945993815,
                    0.8671462352248921,
                    0.0,
                    0.0,
                    0.010936249246006733,
                    4.9724034462308655,
                    0.0,
                    0.000109529143256014,
                    4.767268286781032,
                    5.696583588074738,
                    0.33679636106422556,
                    6.2790357293642245,
                    0.08576556393506203,
                    0.9791867227176846,
                    5.2203636707216745,
                    3.505395331166373,
                    0.16900226789139422,
                    3.233939270644896,
                    0.11958734775694245,
                    0.9939275458750643,
                    1.7888448516768984,
                    0.0,
                    0.011093480506088152,
                    0.0,
                    0.6418795039434974,
                    0.0,
                    0.6297279489280337,
                    0.049265104428481234,
                    0.9906884011621339,
                    2.2838115126365053,
                    0.17661372851070334,
                    0.010017617084974114,
                    1.4206361974718458,
                    0.03238953648207918,
                    0.9995954679949206,
                    0.729608984263009,
                    0.026057904734139103,
                    0.9901206275275505,
                    0.0,
                    0.0,
                    0.0,
                    5.915385743651452,
                    3.1018373837260866,
                    0.022695357887294394,
                    0.09548577139097658,
                    0.0016883448706583705,
                    0.13692274925679673,
                    1.4183254922983506,
                    0.0,
                    0.011506459130503468,
                    2.7270445941308545,
                    5.682229687402327,
                    0.9907936346024356,
                    2.7276023733663592,
                    0.01659430380886691,
                    0.9645108583247105,
                    0.4160275221101905,
                    0.010745413567962719,
                    0.5770530927790579,
                    0.0,
                    4.142714879162745,
                    0.13137383028530206,
                    0.0,
                    5.413827703403013,
                    0.0023114750035640986,
                    0.26089587106309836,
                    5.486692416956518,
                    0.0029336500014880895,
                    2.7920467264674556,
                    4.74253313008952,
                    0.013444908744992348,
                    0.0,
                    0.0,
                    0.004113988431133486,
                    2.920119360618916,
                    0.04227933074480691,
                    0.995198948254196,
                    0.3849986639439985,
                    0.025166485520230836,
                    0.6776228563923579,
                    1.810612821874287,
                    0.0,
                    0.020939107319664044,
                    0.0,
                    2.102414970540268,
                    0.022886697743255816,
                    0.916669093125174,
                    4.377342263285142,
                    0.5461811717094759,
                    6.0832218726653515,
                    2.8766498353947,
                    0.9916655526511906,
                    0.2861522494621492,
                    6.218275003520478,
                    0.7468281469010833,
                    5.5682858360163605,
                    0.26581696369909674,
                    0.5396809735828896,
                    2.3307775291196093,
                    2.2077749235102218,
                    0.0013153198722572923,
                    0.0,
                    0.4884360659169675,
                    0.006496985685736526,
                    3.059290112333984,
                    0.33359221218311164,
                    0.5365979051245621,
                    5.314601460895261,
                    2.782572807543562,
                    0.03423525011697108,
                    0.0,
                    0.0,
                    0.009471586707210657,
                    0.0,
                    2.6294861371303835,
                    0.5994209832540097,
                    0.4457072453586532,
                    4.325784108191396,
                    0.0,
                    0.0,
                    5.968898002548972,
                    0.010353768041659092,
                    0.0,
                    4.308441641533763,
                    0.005445725687574759,
                    6.283185307179586,
                    2.851032592836982,
                    0.0,
                    6.253184535291526,
                    6.2688261187011785,
                    0.03882792858897407,
                    0.8989510539126584,
                    0.19923961321494663,
                    0.986512531406516,
                    0.8500535396836578,
                    0.04164038033496729,
                    0.989595211266979,
                    0.004676311912097243,
                    6.283185307179586,
                    0.034161780440207526,
                    0.00533821460704674,
                    0.0,
                    0.026226115885213373,
                    1.1737920329830305,
                    6.128819512531126,
                    0.9958426960397971,
                    2.850179498931661,
                    6.1641721231730955,
                    0.9707443536603829,
                    5.435724432998263,
                    0.0,
                    0.0014672193260896295,
                    6.283185307179586,
                    6.283185307179586,
                    0.010611074488289527,
                    0.0,
                    0.0,
                    0.011654732639121687,
                    0.0,
                    0.0,
                    0.013899344798857455,
                    1.2296923338242698,
                    6.283185307179586,
                    0.9968959935135387,
                    1.2357826525306141,
                    0.22767649711024907,
                    0.9977399973432458,
                    4.303215207959488,
                    2.848064695077962,
                    1.0,
                    1.1437218554202342,
                    6.283185307179586,
                    0.9987306323271877,
                    1.1407916795396575,
                    0.45965680417383886,
                    1.0,
                    1.9362506044267447,
                    3.3925077280212212,
                    0.0013282631151557548,
                    0.014082995921951292,
                    0.0,
                    0.03269018692722313,
                    0.013522354174553561,
                    0.0,
                    0.03356769889491856,
                    4.704848143855723,
                    0.0,
                    0.0,
                    0.011549491048524901,
                    0.0,
                    0.03362879681470986,
                    6.283185307179586,
                    0.7795319206021721,
                    1.0,
                    0.4239778462086939,
                    0.4647914093782526,
                    0.48956627910751926,
                    0.33812893699887225,
                    2.2375605646588754,
                    0.004467905550154204,
                    0.0,
                    0.0,
                    0.02822724099857087,
                    0.0,
                    0.0,
                    0.02601344472459694,
                    0.0,
                    0.0,
                    0.02354601063029198,
                    5.433622526931153,
                    0.0,
                    0.003544551429925954,
                    1.195800233911284,
                    6.264766930877064,
                    1.0,
                    3.8062703840623637,
                    0.0,
                    0.0,
                    0.11946681039956535,
                    6.270175114754668,
                    0.4628267388515329,
                    0.0,
                    0.0,
                    0.00855074706918823,
                    4.454276059345488,
                    1.7341343467810606,
                    0.04038596226044298,
                    3.919935587337179,
                    0.0,
                    0.0,
                    0.0,
                    5.2133233611009215,
                    0.005627526113709585,
                    0.0,
                    0.215353373349551,
                    0.009138905672518792,
                    3.205758788949745,
                    2.7024306529354254,
                    0.00836826845283117,
                    0.0,
                    0.0,
                    0.01005293490087638,
                    6.283185307179586,
                    0.0,
                    0.010209575518878565,
                    3.996767399283897,
                    0.5770175520799954,
                    0.03193833990851512,
                    0.0,
                    2.1239344845714268,
                    0.9910969374543374,
                    2.931597718023541,
                    0.9577581897129139,
                    0.04462760381106934,
                    0.0,
                    0.0,
                    0.010540837353311941,
                    6.283185307179586,
                    0.0,
                    0.010555106605026485,
                    6.246643004517879,
                    2.284404318949895,
                    0.9901781842827176,
                    0.0,
                    0.0,
                    0.010481178026090414,
                    2.3137921432101085,
                    5.271685600122437,
                    0.0,
                    2.6446650065358153,
                    4.840657391804597,
                    0.0011390735455584088,
                    1.192188797496541,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0], dtype=np.longdouble)


def get_throttle_history(control_states):
    assert isinstance(control_states, np.ndarray)
    assert control_states.ndim == 2
    assert control_states.shape[1] == 3

    throttle_history = np.zeros(control_states.shape[0],)
    for i, control in enumerate(control_states):
        throttle_history[i] = np.linalg.norm(control)

    return throttle_history


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
    # p.set_title(title_in='EJ2000 MJS: {}'.format(epoch), fontsize=20)

    q = LinePlot()
    q.grid()
    q.set_xlabel(xlabel_in=r'X (km)', fontsize=15)
    q.set_ylabel(ylabel_in=r'Z (km)', fontsize=15)
    # q.set_title(title_in='EJ2000 MJS: {}'.format(epoch), fontsize=20)

    epoch, state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000() # at deployment
    if body.name=='Moon': epoch, state = get_rv_state_relative_to_moon(epoch, state)

    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    post_flyby_epoch, state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000() # after flyby around earth
    if body.name=='Moon': post_flyby_epoch, state = get_rv_state_relative_to_moon(post_flyby_epoch, state)
    
    p.plot(xdata=np.array([state[0], ]), ydata=np.array([state[1], ]), color='red', marker='x', markersize=7, linewidth=2)
    q.plot(xdata=np.array([state[0], ]), ydata=np.array([state[2], ]), color='red', marker='x', markersize=7, linewidth=2)

    NRHO_insertion_epoch, state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000() # right before NRHO insertion
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


# generate CR3BP boundary conditions
def generate_CR3BP_boundary_condition():

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    cr3bp = pydylan.eom.CR3BP(primary=earth, secondary=moon)

    libration_point_information_L2 = cr3bp.find_equilibrium_point(pydylan.enum.LibrationPoint.L2)
    desired_orbit_energy_L2 = libration_point_information_L2[1] + 100E-4
    lyapunov_L2 = pydylan.periodic_orbit.Lyapunov(cr3bp, pydylan.enum.LibrationPoint.L2, desired_orbit_energy_L2)
    lyapunov_L2 = periodic_orbit_continuation(lyapunov_L2, desired_orbit_energy_L2)

    L2_manifold_arc = lyapunov_L2.generate_manifold_arc(lyapunov_L2.orbit_period, 4., pydylan.enum.PerturbationDirection.StableRight)

    return pydylan.CR3BPStateBoundaryCondition(earth, moon, L2_manifold_arc.get_end_state(), 1E3, 1E-2)


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


# convert rv state relative to coe state
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


# convert coe state relative to rv state
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
# TODO Need to change to compute a weighted sum
def compute_error_in_state(current_state: np.array, target_state: np.array, w: np.array):

    w = w / np.linalg.norm(w)
    err = w * (current_state - target_state)
    err = np.dot(err, err) / np.dot(target_state, target_state)

    return np.linalg.norm(err)


def read_qlaw_output(qlawhistory):

    # requires qlaw output to be written to "./qlaw_history.csv"
    qlawhistory = pd.read_csv(qlawhistory, names=["epoch", "sma", "ecc", "inc", "ape", "lan", "mass", "tru", "u0", "u1", "u2"])
    epoch, sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2 = qlawhistory.epoch, qlawhistory.sma, qlawhistory.ecc, qlawhistory.inc, qlawhistory.ape, qlawhistory.lan, qlawhistory.tru, qlawhistory.mass, qlawhistory.u0, qlawhistory.u1, qlawhistory.u2

    epoch, sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2 = epoch.to_numpy(), sma.to_numpy(), ecc.to_numpy(), inc.to_numpy(), ape.to_numpy(), lan.to_numpy(), tru.to_numpy() % (2 * np.pi), mass.to_numpy() - thruster_parameters.dry_mass, u0.to_numpy(), u1.to_numpy(), u2.to_numpy() 
    
    states = np.vstack((sma, ecc, inc, ape, lan, tru, mass, u0, u1, u2)).T
    states = get_rv_state_from_coe_state(pydylan.Body("Moon"), states)

    return epoch, states


# obtain the trajectory from the NRHO_ORBIT to a LOW_LUNAR_ORBIT using a qlaw method
def solve_qlaw(initial_epoch, initial_state, thruster_parameters):

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
    settings.t0 = initial_epoch
    # settings.m_petro = 8.772
    # settings.n_petro = 4.209
    # settings.r_petro = 2.502
    # settings.k_petro = 7.378    
    # settings.b_petro = 0.01
    # settings.minimum_periapse = 100

    qlawsolver = pydylan.qlaw.QLaw5_Solver(moon, coe_i, coe_f, sc, settings)

    print(f"INITIAL COE -> sma: {initial_state[0]} km, ecc: {initial_state[1]}, inc: {initial_state[2] * RAD2DEG}, arg_per: {initial_state[3] * RAD2DEG}, lon_asc: {initial_state[4] * RAD2DEG}, tru_ano: {(initial_state[5] * RAD2DEG) % 360}")

    final_state = qlawsolver.solve()

    sma = final_state[0] / (1 - final_state[1] * final_state[1]) * moon.DU
    ecc = final_state[1]
    inc = final_state[2] * 180 / np.pi

    print(f"FINAL COE -> sma: {sma} km, ecc: {ecc}, inc: {final_state[2] * RAD2DEG}, arg_per: {final_state[3] * RAD2DEG}, lon_asc: {final_state[4] * RAD2DEG}, tru_ano: {(final_state[6] * RAD2DEG) % 360}")

    assert(qlawsolver.status == pydylan.qlaw.QLaw5_Solver_Status.converged)
    assert(abs(sma - coe_f.sma) < settings.tol_sma)
    assert(abs(ecc - coe_f.ecc) < settings.tol_ecc)
    assert(abs(inc - coe_f.inc) < settings.tol_inc)

    epoch, states = read_qlaw_output(qlawhistory)

    return epoch, states


# obtain the trajectory from the NRHO_INSERTION_POINT to a NRHO_ORBIT by thrusting in the velocity direction
def solve_spiral(eom: pydylan.eom, thruster_parameters:pydylan.ThrustParameters, initial_epoch, initial_state, verbose=True) -> (np.ndarray, np.ndarray):

    moon = pydylan.Body("Moon")

    initial_epoch, initial_state = get_rv_state_relative_to_moon(initial_epoch, initial_state)
    current_epoch, current_state = initial_epoch, initial_state

    error_weight = np.array([0.45, 0.1, 0.45, 0., 0., 0.])
    
    target_coe_state = np.array([4287 * 2.0, 0.5714, np.pi / 2, 0., 0., 0.])
    current_coe_state = get_coe_state_from_rv_state(moon, current_state)
    current_error = compute_error_in_state(current_coe_state, target_coe_state, error_weight)

    print("Current COE state: ", current_coe_state)
    print("Target COE state: ", target_coe_state)

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

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.FixedBoundaryCondition(middle_state)
    right_boundary_condition = pydylan.FixedBoundaryCondition(target_state)

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1 #60 * 60 * 1

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    mbh_options.max_step_size = 1.
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1 #60 * 60 * 6

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

    output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()
    # np.save("./Plots_for_LIC/lic_lt_feasible_control_solutions_full_fixed_BC.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states


# obtain the trajectory from the POST_DEPLOYMENT_STATE to NRHO_INSERTION_STATE with Fixed+Advecting+CR3BP BCs
def solve_transfer_other_BC(eom, thruster_parameters) -> (np.ndarray, np.ndarray):

    initial_epoch, initial_state = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    middle_epoch, middle_state = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    target_epoch, target_state = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()

    left_boundary_condition = pydylan.FixedBoundaryCondition(initial_state)
    middle_boundary_condition = pydylan.AdvectingBoundaryCondition(middle_state, middle_epoch, eom)
    right_boundary_condition = generate_CR3BP_boundary_condition()

    snopt_options = pydylan.SNOPT_options_structure()
    snopt_options.derivative_mode = pydylan.enum.derivative_mode_type.analytic
    snopt_options.solver_mode = pydylan.enum.solver_mode_type.feasible
    snopt_options.quiet_SNOPT = False
    snopt_options.time_limit = 1 #60 * 60 * 1

    mbh_options = pydylan.MBH_options_structure()
    mbh_options.hop_mode = pydylan.enum.mbh_hop_mode_type.hop
    # mbh_options.max_step_size = 1.
    mbh_options.quiet_MBH = False
    mbh_options.time_limit = 1 #60 * 60 * 6

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

    output_control_to_screen(mission.get_control_state())
    
    assert mission.is_best_solution_feasible()
    # np.save("./Plots_for_LIC/lic_lt_feasible_control_solutions_full_other_BC.npy", mission.get_all_feasible_control_solutions())

    result = mission.evaluate_and_return_solution(mission.get_control_state(), forward_backward_shooting=True)

    time = result.time
    states = result.states

    return time, states


if __name__ == '__main__':

    zoomed_in = False # For visuals
    
    # Load SPICE kernels
    kernels = pydylan.spice.load_spice()

    earth = pydylan.Body("Earth")
    moon = pydylan.Body("Moon")
    sun = pydylan.Body("Sun")
    eom_earth = pydylan.eom.Ephemeris_nBP(earth)
    eom_earth.add_secondary_body(moon)
    eom_earth.add_secondary_body(sun)
    eom_moon = pydylan.eom.Ephemeris_nBP(moon)
    eom_moon.add_secondary_body(earth)
    eom_moon.add_secondary_body(sun)

    initial_epoch, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()

    thruster_parameters = pydylan.ThrustParameters(fuel_mass=1.5, dry_mass=12.5, Isp=2500., thrust=1.5E-3)

    time_transfer, states_transfer = solve_transfer_other_BC(eom_earth, thruster_parameters)

    thruster_parameters.fuel_mass = states_transfer[-1,6]
    time_spiral, states_spiral = solve_spiral(eom_moon, thruster_parameters, time_transfer[-1], states_transfer[-1, :6])

    thruster_parameters.fuel_mass = states_spiral[-1,6]
    time_qlaw, states_qlaw = solve_qlaw(time_spiral[-1], states_spiral[-1, :6], thruster_parameters)

    _, states_transfer_rel_moon = get_rv_state_relative_to_moon(time_transfer, states_transfer)
    _, states_spiral_rel_earth = get_rv_state_relative_to_earth(time_spiral, states_spiral)
    _, states_qlaw_rel_earth = get_rv_state_relative_to_earth(time_qlaw, states_qlaw)

    p, q = get_plots_of_phase_integration(earth, eom_earth)
    p.plot(xdata=states_transfer[:, 0], ydata=states_transfer[:, 1], color='black', linewidth=1)
    q.plot(xdata=states_transfer[:, 0], ydata=states_transfer[:, 2], color='black', linewidth=1)
    p.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 1], color=strongorange, linewidth=1, alpha=0.9)
    q.plot(xdata=states_spiral_rel_earth[:, 0], ydata=states_spiral_rel_earth[:, 2], color=strongorange, linewidth=1, alpha=0.9)
    p.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 1], color=babyblue, linewidth=1, alpha=0.8)
    q.plot(xdata=states_qlaw_rel_earth[:, 0], ydata=states_qlaw_rel_earth[:, 2], color=babyblue, linewidth=1, alpha=0.8)

    # # Mark the end of the spiral phase
    # p.plot(xdata=np.array([states_spiral_rel_earth[-1, 0], ]), ydata=np.array([states_spiral_rel_earth[-1, 1], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # q.plot(xdata=np.array([states_spiral_rel_earth[-1, 0], ]), ydata=np.array([states_spiral_rel_earth[-1, 2], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # # Mark the beginning of the qlaw phase
    # p.plot(xdata=np.array([states_qlaw_rel_earth[0, 0], ]), ydata=np.array([states_qlaw_rel_earth[0, 1], ]), color='green', marker='x', markersize=10, linewidth=2)
    # q.plot(xdata=np.array([states_qlaw_rel_earth[0, 0], ]), ydata=np.array([states_qlaw_rel_earth[0, 2], ]), color='green', marker='x', markersize=10, linewidth=2)

    r, s = get_plots_of_phase_integration(moon, eom_moon)
    r.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 1], color='black', linewidth=1)
    s.plot(xdata=states_transfer_rel_moon[:, 0], ydata=states_transfer_rel_moon[:, 2], color='black', linewidth=1)
    r.plot(xdata=states_spiral[:, 0], ydata=states_spiral[:, 1], color=strongorange, linewidth=1, alpha=0.9)
    s.plot(xdata=states_spiral[:, 0], ydata=states_spiral[:, 2], color=strongorange, linewidth=1, alpha=0.9)
    r.plot(xdata=states_qlaw[:, 0], ydata=states_qlaw[:, 1], color=babyblue, linewidth=1, alpha=0.7)
    s.plot(xdata=states_qlaw[:, 0], ydata=states_qlaw[:, 2], color=babyblue, linewidth=1, alpha=0.7)

    # # Mark the end of the spiral phase
    # r.plot(xdata=np.array([states_spiral[-1, 0], ]), ydata=np.array([states_spiral[-1, 1], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # s.plot(xdata=np.array([states_spiral[-1, 0], ]), ydata=np.array([states_spiral[-1, 2], ]), color='blue', marker='x', markersize=10, linewidth=2)
    # # Mark the beginning of the qlaw phase
    # r.plot(xdata=np.array([states_qlaw[0, 0], ]), ydata=np.array([states_qlaw[0, 1], ]), color='green', marker='x', markersize=10, linewidth=2)
    # s.plot(xdata=np.array([states_qlaw[0, 0], ]), ydata=np.array([states_qlaw[0, 2], ]), color='green', marker='x', markersize=10, linewidth=2)

    if zoomed_in==True:
        r.set_axis('tight', [-12000, 12000, -20000, 50000])
        s.set_axis('tight', [-50000, 50000, -60000, 25000])

    d1, _ = get_post_deployment_epoch_in_MJS_and_state_in_EJ2000()
    d1 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d1)
    d2, _ = get_post_flyby_epoch_in_MJS_and_state_in_EJ2000()
    d2 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d2)
    d3, _ = get_NRHO_insertion_epoch_in_MJS_and_state_in_EJ2000()
    d3 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d3)
    d4 = time_spiral[-1]
    d4 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d4)
    d5 = time_qlaw[-1]
    d5 = pydylan.epoch.convert_relative_MJS_to_YMDHMS(d5)

    time_transfer, time_spiral, time_qlaw = (time_transfer - initial_epoch) * SEC2DAY, (time_spiral - initial_epoch) * SEC2DAY, (time_qlaw - initial_epoch) * SEC2DAY
    t = LinePlot()
    t.grid()
    t.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t.plot(xdata=time_transfer, ydata=get_throttle_history(states_transfer[:, -3:]), color='black', linewidth=2)

    t_spiral = LinePlot()
    t_spiral.grid()
    t_spiral.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t_spiral.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t_spiral.plot(xdata=time_spiral, ydata=get_throttle_history(states_spiral[:, -3:]), color=strongorange, linewidth=2)

    t_qlaw = LinePlot()
    t_qlaw.grid()
    t_qlaw.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    t_qlaw.set_ylabel(ylabel_in=r'Throttle', fontsize=15)
    t_qlaw.plot(xdata=time_qlaw, ydata=get_throttle_history(states_qlaw[:, -3:]), color=babyblue, linewidth=2)

    m = LinePlot()
    m.grid()
    m.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m.plot(xdata=time_transfer, ydata=states_transfer[:, 6], color='black', linewidth=2)

    m_spiral = LinePlot()
    m_spiral.grid()
    m_spiral.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m_spiral.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m_spiral.plot(xdata=time_spiral, ydata=states_spiral[:, 6], color=strongorange, linewidth=2)

    m_qlaw = LinePlot()
    m_qlaw.grid()
    m_qlaw.set_xlabel(xlabel_in=r'Time (days)', fontsize=15)
    m_qlaw.set_ylabel(ylabel_in=r'Mass (kg)', fontsize=15)
    m_qlaw.plot(xdata=time_qlaw, ydata=states_qlaw[:, 6], color=babyblue, linewidth=2)

    fig, axs = plt.subplots(3, 2, sharex=True)
    axs[0, 0].plot(time_transfer, states_transfer[:, 0])
    axs[0, 0].plot(time_spiral, states_spiral_rel_earth[:, 0], color=strongorange)
    axs[0, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 0], color=babyblue)
    axs[0, 0].set_title('X-coordinate')
    axs[0, 0].grid()
    axs[1, 0].plot(time_transfer, states_transfer[:, 1])
    axs[1, 0].plot(time_spiral, states_spiral_rel_earth[:, 1], color=strongorange)
    axs[1, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 1], color=babyblue)
    axs[1, 0].set_title('Y-coordinate')
    axs[1, 0].grid()
    axs[2, 0].plot(time_transfer, states_transfer[:, 2])
    axs[2, 0].plot(time_spiral, states_spiral_rel_earth[:, 2], color=strongorange)
    axs[2, 0].plot(time_qlaw, states_qlaw_rel_earth[:, 2], color=babyblue)
    axs[2, 0].set_title('Z-coordinate')
    axs[2, 0].grid()
    axs[0, 1].plot(time_transfer, states_transfer[:, 3])
    axs[0, 1].plot(time_spiral, states_spiral_rel_earth[:, 3], color=strongorange)
    axs[0, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 3], color=babyblue)
    axs[0, 1].set_title('VX-coordinate')
    axs[0, 1].grid()
    axs[1, 1].plot(time_transfer, states_transfer[:, 4])
    axs[1, 1].plot(time_spiral, states_spiral_rel_earth[:, 4], color=strongorange)
    axs[1, 1].plot(time_qlaw, states_qlaw_rel_earth[:, 4], color=babyblue)
    axs[1, 1].set_title('VY-coordinate')
    axs[1, 1].grid()
    axs[2, 1].plot(time_transfer, states_transfer[:, 5])
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

    print('\n Mass Remaining at the end of NRHO-INSERTION', states_transfer[-1, 6], ' kg')
    print('\n Mass Remaining at the end of SPIRAL', states_spiral[-1, 6], ' kg')
    print('\n Mass Remaining at the end of QLAW', states_qlaw[-1, 6], ' kg')

    plt.show()

    # Unload SPICE kernels
    pydylan.spice.unload_spice(kernels)