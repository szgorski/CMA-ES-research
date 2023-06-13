
from enum import Enum
from typing import List, Tuple, Dict
import numpy as np
import statistics


class Algorithm(Enum):
    CMAES = 1,
    MAES = 2,
    IPOP = 3


FUNCTION_OPTIMAL_VALUES: Dict = {
    "F10: Composition Function 3 (F24 CEC-2017)": 2500,
    "F1: Shifted and Rotated Bent Cigar Function (F1 CEC-2017)": 100,
    "F2: Shifted and Rotated Schwefel’s Function (F11 CEC-2014)": 1100,
    "F3: Shifted and Rotated Lunacek bi-Rastrigin Function (F7 CEC-2017)": 700,
    "F4: Expanded Rosenbrock’s plus Griewangk’s Function (F15 CEC-2014)": 1900,
    "F17: Hybrid Function 1 (F17 CEC-2014)": 1700,
    "F16: Hybrid Function 2 (F15 CEC-2017)": 1600,
    "F7: Hybrid Function 3 (F21 CEC-2014)": 2100,
    "F8: Composition Function 1 (F21 CEC-2017)": 2200,
    "F9: Composition Function 2 (F23 CEC-2017)": 2400
}


def calculate_normalized_error(f_best: float, functionOptimalValue: float, f_best_max: float) -> float:
    return (f_best - functionOptimalValue) / (f_best_max - functionOptimalValue)


def get_f_best_max(packets: List[Tuple[str, List[float]]], functionName: str) -> float:
    functionValues = next(x[1] for x in packets if x[0] == functionName)
    return max(functionValues)


def get_f_best_min(packets: List[Tuple[str, List[float]]], functionName: str) -> float:
    functionValues = next(x[1] for x in packets if x[0] == functionName)
    return min(functionValues)


def get_f_best_max_of_all_algorithms(algorithmsPackets: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], functionName: str) -> float:
    maxValue = float('-inf')
    for packet in algorithmsPackets:
        calculatedBestValue = get_f_best_max(packet[1], functionName)
        if calculatedBestValue > maxValue:
            maxValue = calculatedBestValue

    return maxValue


def calculate_score_1(allPackets10D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], allPackets20D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]]):
    algorithms_sne = [get_SNE_by_algorithm(
        allPackets10D, allPackets20D, algorithm) for algorithm in [e.value for e in Algorithm]]

    return (1 - (sum(algorithms_sne)-min(algorithms_sne))/sum(algorithms_sne)) * 50


def get_normalized_errors_sum_for_given_algorithm_by_packets(allPackets: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], algorithm: Algorithm) -> float:
    algoPackets = next(x for x in allPackets if x[0].value == algorithm)
    normlaized_errors_sum = 0

    for function in list(FUNCTION_OPTIMAL_VALUES.keys()):
        f_best_max = get_f_best_max_of_all_algorithms(
            allPackets, function)

        # functionValuesOfalgorithm = next(x[1] for x in algoPacket[1] if x[0] == function)
        f_best_min = get_f_best_min(algoPackets[1], function)

        normalized_error = calculate_normalized_error(
            f_best_min, FUNCTION_OPTIMAL_VALUES[function], f_best_max)
        normlaized_errors_sum += normalized_error

    return normlaized_errors_sum


def get_rank_of_algorithms_by_packets(allPackets: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]]) -> float:
    # [CMAES, MAES, IPOP] ranks
    ranks_values_for_algorithms: List[float] = [0, 0, 0]
    rank_points = [0, 0, 0]

    for function in list(FUNCTION_OPTIMAL_VALUES.keys()):
        median_values_for_function = [0, 0, 0]

        for i in range(len(allPackets)):
            functionValues = next(x[1]
                                  for x in allPackets[i][1] if x[0] == function)
            median_values_for_function[i] = statistics.median(functionValues)

    # situation when we have multiple same values
        if len(median_values_for_function) != len(set(median_values_for_function)):
            median_values_set = list(set(median_values_for_function))
            median_values_set.sort()

            for i in range(len(median_values_for_function)):
                for j in range(len(median_values_set)):
                    if median_values_set[j] == median_values_for_function[i]:
                        rank_points[i] += j+1

        else:
            rank_points = [x+1 for x in np.argsort(median_values_for_function)]

        for i in range(len(ranks_values_for_algorithms)):
            ranks_values_for_algorithms[i] += rank_points[i]

    return [i*0.5 for i in ranks_values_for_algorithms]


def get_SR_of_alogorithms(allPackets10D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], allPackets20D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]]) -> List[float]:
    ranks10D = get_rank_of_algorithms_by_packets(allPackets10D)
    ranks20D = get_rank_of_algorithms_by_packets(allPackets20D)
    result = np.zeros(len(ranks10D))

    for i in range(len(ranks10D)):
        result[i] = ranks10D[i]+ranks20D[i]

    return result


def calculate_score_2(allPackets10D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], allPackets20D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]]):
    algorithms_sr = get_SR_of_alogorithms(allPackets10D, allPackets20D)

    return (1 - (sum(algorithms_sr)-min(algorithms_sr))/sum(algorithms_sr)) * 50


def get_SNE_by_algorithm(allPackets10D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], allPackets20D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], algorithm: Algorithm):
    return 0.5*get_normalized_errors_sum_for_given_algorithm_by_packets(allPackets10D, algorithm) + 0.5*get_normalized_errors_sum_for_given_algorithm_by_packets(allPackets20D, algorithm)


def calculate_algorithms_score(allPackets10D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]], allPackets20D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]]):
    return calculate_score_1(allPackets10D, allPackets20D) + calculate_score_2(allPackets10D, allPackets20D)


def main():
    # 1 -packet - 10 functions , for each function 30 results, overall six packets
    cmaes_packet_10D: Tuple[Algorithm, List[Tuple[str, List[float]]]] = (Algorithm.CMAES, [('F10: Composition Function 3 (F24 CEC-2017)', [4080.787255603214, 4152.514337373978, 4085.745632116298, 4211.227564408982, 3986.0726725675195, 4051.114318619524, 4143.318729715267, 3915.7158142648645, 4174.810256613088, 4003.241685060596, 4151.8420330644085, 4079.460616061615, 4137.193632361295, 4136.859424812259, 3974.110515038437, 4127.998346303924, 4090.4121592611664, 4064.3102638005735, 3955.072223286944, 4162.344773923116, 4039.4396517350397, 4111.182334446337, 4077.856817737237, 4073.5410187069074, 4067.0875549934353, 4181.9788059231, 3957.9067278609486, 4101.266953871949, 4147.561807286672, 4098.733759919407]), ('F1: Shifted and Rotated Bent Cigar Function (F1 CEC-2017)', [26351421566.555473, 26325716900.977123, 26558800940.642437, 27384486384.942333, 27651151132.026268, 27885314593.489887, 26993797916.703682, 26086463848.035206, 27225518896.78102, 27036744920.03489, 27209372512.993294, 26879923337.316925, 26785256603.69922, 26794722441.040718, 27862498163.937035, 26357376341.97945, 27239117257.595036, 27751658902.05174, 27261163104.146366, 23681041316.35323, 27204390573.26339, 25409030236.151596, 26015551245.947655, 25303623041.423805, 26406461111.12949, 27247082327.31748, 27879834286.006897, 27789587877.336277, 26408415703.958122, 27621876530.807346]), ('F2: Shifted and Rotated Schwefel’s Function (F11 CEC-2014)', [2436624332739.693, 2400402967156.0166, 2317073268820.1865, 2421649932788.4976, 2289433078558.774, 2364268903533.78, 2385101118208.569, 2443031766248.2476, 2426999319705.7744, 2386804993218.4136, 2466632086328.2314, 2490060815797.1733, 2545614198983.3994, 2548279941959.3164, 2452154310779.4033, 2394439539665.25, 2494916334128.073, 2472964172913.2334, 2346617826717.66, 2525256319024.2773, 2372436434472.418, 2241682850571.4326, 2345393227780.594, 2540997153515.191, 2441245841805.309, 2429977168937.9316, 2490479003131.411, 2518416191142.7437, 2427330528259.3984, 2499172262966.245]), ('F3: Shifted and Rotated Lunacek bi-Rastrigin Function (F7 CEC-2017)', [497820890719.6719, 499855563968.65155, 482898846753.82025, 524047202919.8839, 496497555766.4098, 502293394744.7212, 523552887613.7604, 490932001951.7206, 501673132808.25836, 496227906898.6332, 527669511854.6708, 479989953953.1803, 529523471984.7306, 514090591648.6835, 492712100450.68353, 492057796713.5485, 500471773319.59625, 522298340905.28296, 503299658246.4028, 516767481704.5, 487557661797.19604, 528149014415.67163, 510090117372.51917, 524271581315.37494, 489831908480.1174, 537423675783.2576, 501322329258.53503, 510674106186.46246, 503898665285.23114, 474720635348.9961]), ('F4: Expanded Rosenbrock’s plus Griewangk’s Function (F15 CEC-2014)', [1095147.873668295, 1171420.7225006719, 681257.2178575395, 666055.9287979616, 889226.1119331134, 1165924.0183305643, 780473.8879707038, 830787.3160514656, 872825.0584353283, 1345444.8310337448, 1199560.8808918227, 1055462.8147674883, 1143586.3611618215, 1019775.0054651804, 1190998.177632401, 1105880.8586127367, 1292446.4228915195, 1256372.9134855072, 1069605.4231431966, 937600.0376588993, 970232.6028529779, 1049151.667830408, 1202134.1777209975, 1011173.9308809923, 952802.8405537973, 975886.38436906,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               827902.7208604122, 1394826.3709530982, 1214299.7866663137, 785826.3917948807]), ('F17: Hybrid Function 1 (F17 CEC-2014)', [235929912.17219806, 235711840.13377884, 225979190.45033163, 306901705.32549256, 165357517.37947935, 299814693.52856237, 293944185.22224313, 248725706.97747758, 177093411.59631097, 301662104.2197301, 101661610.23931871, 239774693.79707354, 160886342.66241875, 288460652.7036839, 306214357.48171914, 280073275.5071293, 321156375.9850368, 212260525.32313678, 267089041.28976616, 362079833.94699997, 292193788.5571804, 202003410.40805575, 241008519.65158582, 205258255.71853036, 371101319.4504448, 307667229.9991396, 122270739.77436763, 281697001.7165097, 311905667.66478884, 200182893.34157336]), ('F16: Hybrid Function 2 (F15 CEC-2017)', [6431025359.239888, 6130083501.662782, 5921383840.614282, 5157784592.077013, 5118845423.9924755, 5898470869.669817, 5932941969.744688, 4680056818.450927, 6359636407.625159, 5487182834.143923, 6095290031.03678, 6020157727.923406, 5896259916.368622, 6652209482.173509, 6545010634.312313, 5936598236.713679, 6405003191.198464, 5861175810.101396, 5394161411.749737, 6549081001.988856, 5125164439.182655, 5817511623.040659, 6223962178.665645, 6413900704.024728, 5579733769.595516, 5747375596.696425, 6405996744.141007, 6000085359.178422, 5786255603.028996, 5436876427.479668]), ('F7: Hybrid Function 3 (F21 CEC-2014)', [607535379.9749455, 643289754.5165807, 483274228.07816666, 744002732.3995217, 576815742.2028027, 782382289.3776784, 792539179.0946074, 766461402.8997285, 558493646.4620132, 667184585.5719, 703602604.7135634, 682273389.1359063, 723669032.0723603, 836985515.4084079, 894620685.0501412, 690203795.1764649, 890556605.8886158, 688404304.3787534, 620430305.1015301, 862424955.695786, 800344851.3644654, 593473877.1728387, 731078369.5528268, 672937920.746623, 943695651.1650639, 885133063.67388, 453320690.42882097, 770959818.0353488, 700143896.3775847, 695600432.9568968]), ('F8: Composition Function 1 (F21 CEC-2017)', [4021.949347517206, 4163.176961352743, 4051.9079938926375, 4106.04220937988, 3953.082641343609, 3878.000859998898, 4021.4057115272135, 3987.8070639120165, 4009.4137365710185, 4155.369921408356, 3915.4535715485367, 4013.8928582844574, 4050.2915087508, 3801.6775391295787, 4161.9794730597805, 3939.0906214568604, 4051.6550079915937, 4035.145539559937, 3866.291764084499, 3942.5536322903317, 4034.917217254705, 4032.7280293130275, 4102.704643716857, 3972.524560146094, 3976.709921059068, 4128.371472011711, 4118.053543979478, 4146.046297085879, 4167.841713958108, 4206.478763708641]), ('F9: Composition Function 2 (F23 CEC-2017)', [11427.812725513875, 12092.847063818243, 11642.29167528914, 11928.915207328839, 11266.114532667225, 11570.443921483684, 12208.57606840162, 12058.8144163183, 12017.473785239345, 12036.264991055383, 11764.605741619285, 11803.217930767602, 11943.107640624958, 11993.94999898118, 11703.012803337426, 11357.00447175978, 12226.502324608056, 11930.65268440463, 12037.22916637081, 10999.019343837574, 11879.52539015045, 12132.49915528271, 11816.866871482955, 11928.085761260225, 11692.281915780259, 11998.83778663803, 12025.961666882265, 12053.970221823896, 12186.974322559492, 12270.902764952161])])

    # 3 packets for 10D
    packets_10D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]] = [
        cmaes_packet_10D, (Algorithm.MAES, cmaes_packet_10D[1]), (Algorithm.IPOP, cmaes_packet_10D[1])]
    # 3 packets for 20D
    packets_20D: List[Tuple[Algorithm, List[Tuple[str, List[float]]]]] = [
        cmaes_packet_10D, (Algorithm.MAES, cmaes_packet_10D[1]), (Algorithm.IPOP, cmaes_packet_10D[1])]

    print(calculate_algorithms_score(packets_10D, packets_20D))
    pass


main()
