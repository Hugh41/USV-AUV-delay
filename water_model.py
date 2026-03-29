import numpy as np
from scipy.special import erfc
# 这个文件通过规范新的模型，从而实现当噪声很大情况下的有效模拟。当然，在这种
# 情况下，也许对于SNR本身，也要进行显著降低？


frequency_khz = 20          # 信号频率 (kHz)
source_level_db = 135       # 声源级 SL (dB re 1μPa)  155.54121254719664 => 3e-2W
noise_level_db = 87         # 环境噪声级 NL (dB) —— 可根据Wenz模型细化  默认85
directivity_index_db = 0    # 接收指向性增益 DI (dB)，单水听器=0
sound_speed_mps = 1500
bits_per_packet = 4096

enable_fading = True
fading_sigma = 1 / np.sqrt(2)          # 瑞利衰落参数（sigma），控制衰落深度

def absorption_db_km(freq_khz):
    # Thorp's formula for absorption in dB/km
    if freq_khz < 0.4:
        return 0.003
    elif freq_khz < 50:
        return 0.11 * (freq_khz / 1000) ** 2 / (1 + freq_khz / 1000)
    else:
        return 4.9 * (freq_khz / 1000) ** 2
    
alpha_db_km = absorption_db_km(frequency_khz)
def calculate_transmission_loss(d_m, alpha_db_km):
    """计算传播损失 TL = 球面扩展 + 吸收"""
    d_km = d_m / 1000.0
    spreading_loss = 20 * np.log10(d_m)  # 球面扩展 (20log10(d))
    absorption_loss = alpha_db_km * d_km # 吸收损失
    return spreading_loss + absorption_loss

def calculate_snr(sl_db, tl_db, nl_db, di_db):
    """计算接收信噪比 SNR (dB)"""
    return sl_db - tl_db - nl_db + di_db

def snr_to_ber(snr_linear, modulation="QPSK"):
    """根据调制方式计算理论BER"""
    if modulation.upper() == "BPSK":
        return 0.5 * erfc(np.sqrt(snr_linear))
    elif modulation.upper() == "QPSK":
        return 0.5 * erfc(np.sqrt(snr_linear / 2))
    else:
        raise ValueError("Unsupported modulation. Use 'BPSK' or 'QPSK'.")
    
def ber_to_per(ber, n_bits):
    """计算误包率 PER = 1 - (1 - BER)^n"""
    return 1 - (1 - ber) ** n_bits

def get_package_loss(distance_m):
    '''开始仿真，入参为距离，出参为丢包率'''
    all_per_d = 0; try_time = 500
    for _ in range(try_time):
        tl_db = calculate_transmission_loss(distance_m, alpha_db_km)
        snr_db = calculate_snr(source_level_db, tl_db, noise_level_db, directivity_index_db)
        snr_linear = 10 ** (snr_db / 10)
        if enable_fading:
            h = np.random.rayleigh(fading_sigma)  # 瑞利衰落系数
            snr_faded_linear = snr_linear * (h ** 2)
            snr_faded_db = 10 * np.log10(snr_faded_linear)
            # print(f"衰落后的SNR: {snr_faded_db:.2f} dB")
            snr_used_linear = snr_faded_linear
        else:
            snr_used_linear = snr_linear
        ber = snr_to_ber(snr_used_linear)
        per = ber_to_per(ber, bits_per_packet)
        packet_loss_rate = per
        all_per_d += packet_loss_rate / try_time
    return all_per_d

