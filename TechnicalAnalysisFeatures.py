import numpy
import pandas as pd

def rolling_mean(series, window): return series.rolling(window).mean()
def rolling_std(series, window): return series.rolling(window).std()
def rolling_sum(series, window): return series.rolling(window).sum()
def ewma(series, span, min_periods): return series.ewm(span = span, min_periods = min_periods).mean()
def get_value(df, idx, col): return df.iloc[idx][col]

#Moving Average
def MA(df, n):
    name = 'MA_' + str(n)
    MA = pd.Series(rolling_mean(df['Close'], n), name = name)
    df[name] = MA
    return df

#Exponential Moving Average
def EMA(df, n):
    name = 'EMA_' + str(n)
    EMA = pd.Series(
        ewma(df['Close'], span = n, min_periods = n - 1), 
        name = name)
    df[name] = EMA
    return df

#Momentum
def MOM(df, n):
    name = 'Momentum_' + str(n)
    M = pd.Series(df['Close'].diff(n), name = name)
    df[name] = M
    return df

#Rate of Change
def ROC(df, n):
    name = 'ROC_' + str(n)
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name = name)
    df[name] = ROC
    return df

#Average True Range
def ATR(df, n):
    name = 'ATR_' + str(n)
    i = 0
    TR_l = [0]
    while i < len(df) - 1:
        TR = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(ewma(TR_s, span = n, min_periods = n), name = name)
    df[name] = ATR
    return df

#Bollinger Bands
def BBANDS(df, n):
    name_B1 = 'BollingerB_' + str(n)
    name_B2 = 'Bollinger%b_' + str(n)
    MA = pd.Series(rolling_mean(df['Close'], n))
    MSD = pd.Series(rolling_std(df['Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name = name_B1)
    df[name_B1] = B1
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name = name_B2)
    df[name_B2] = B2
    return df

#Pivot Points, Supports and Resistances
def PPSR(df):
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)
    R1 = pd.Series(2 * PP - df['Low'])
    S1 = pd.Series(2 * PP - df['High'])
    R2 = pd.Series(PP + df['High'] - df['Low'])
    S2 = pd.Series(PP - df['High'] + df['Low'])
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}
    PSR = pd.DataFrame(psr)
    for col in PSR.columns:
        df['PSR_' + col] = PSR[col]
    return df

#Stochastic oscillator %K
def STOK(df):
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name = 'SO%k')
    df['SOk'] = SOk
    return df

# Stochastic Oscillator, EMA smoothing, nS = slowing (1 if no slowing)
def STO(df,  nK, nD, nS=1):
    SOk = pd.Series((df['Close'] - df['Low'].rolling(nK).min()) / (df['High'].rolling(nK).max() - df['Low'].rolling(nK).min()), name = 'SO%k'+str(nK))
    SOd = pd.Series(SOk.ewm(ignore_na=False, span=nD, min_periods=nD-1, adjust=True).mean(), name = 'SO%d'+str(nD))
    SOk = SOk.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()
    SOd = SOd.ewm(ignore_na=False, span=nS, min_periods=nS-1, adjust=True).mean()
    df['SOk'] = SOk
    df['SOd'] = SOd
    return df

#Trix
def TRIX(df, n):
    name = 'Trix_' + str(n)
    EX1 = ewma(df['Close'], span = n, min_periods = n - 1)
    EX2 = ewma(EX1, span = n, min_periods = n - 1)
    EX3 = ewma(EX2, span = n, min_periods = n - 1)
    i = 0
    ROC_l = [0]
    while i + 1 <= len(df) - 1:
        ROC = (EX3.iloc[i + 1] - EX3.iloc[i]) / EX3.iloc[i]
        ROC_l.append(ROC)
        i = i + 1
    Trix = pd.Series(ROC_l, name = name)
    df[name] = Trix
    return df

#Average Directional Movement Index
def ADX(df, n, n_ADX):
    name = 'ADX_' + str(n) + '_' + str(n_ADX)
    i = 0
    UpI = []
    DoI = []
    while i + 1 <= len(df) - 1:
        UpMove = get_value(df, i + 1, 'High') - get_value(df, i, 'High')
        DoMove = get_value(df, i, 'Low') - get_value(df, i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    i = 0
    TR_l = [0]
    while i < len(df) - 1:
        TR = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR_l.append(TR)
        i = i + 1
    TR_s = pd.Series(TR_l)
    ATR = pd.Series(ewma(TR_s, span = n, min_periods = n))
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(ewma(UpI, span = n, min_periods = n - 1) / ATR)
    NegDI = pd.Series(ewma(DoI, span = n, min_periods = n - 1) / ATR)
    ADX = pd.Series(
        ewma(abs(PosDI - NegDI) / (PosDI + NegDI), 
        span = n_ADX, min_periods = n_ADX - 1), name = name)
    df[name] = ADX
    return df

#MACD, MACD Signal and MACD difference
def MACD(df, n_fast, n_slow):
    name_MACD = 'MACD_' + str(n_fast) + '_' + str(n_slow)
    name_MACDsign = 'MACDsign_' + str(n_fast) + '_' + str(n_slow)
    name_MACDdiff = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow)
    EMAfast = pd.Series(ewma(df['Close'], span = n_fast, min_periods = n_slow - 1))
    EMAslow = pd.Series(ewma(df['Close'], span = n_slow, min_periods = n_slow - 1))
    MACD = pd.Series(
        EMAfast - EMAslow, 
        name = 'MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(
        ewma(MACD, span = 9, min_periods = 8), 
        name = 'MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(
        MACD - MACDsign, 
        name = 'MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df[name_MACD] = MACD
    df[name_MACDsign] = MACDsign
    df[name_MACDdiff] = MACDdiff
    return df

#Mass Index
def MassI(df):
    Range = df['High'] - df['Low']
    EX1 = ewma(Range, span = 9, min_periods = 8)
    EX2 = ewma(EX1, span = 9, min_periods = 8)
    Mass = EX1 / EX2
    MassI = pd.Series(rolling_sum(Mass, 25), name = 'Mass Index')
    df['MassI'] = MassI
    return df

#Vortex Indicator: http://www.vortexindicator.com/VFX_VORTEX.PDF
def Vortex(df, n):
    name = 'Vortex_' + str(n)
    i = 0
    TR = [0]
    while i < len(df) - 1:
        Range = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR.append(Range)
        i = i + 1
    i = 0
    VM = [0]
    while i < len(df) - 1:
        Range = abs(get_value(df, i + 1, 'High') - get_value(df, i, 'Low')) - abs(get_value(df, i + 1, 'Low') - get_value(df, i, 'High'))
        VM.append(Range)
        i = i + 1
    VI = pd.Series(
        rolling_sum(pd.Series(VM), n) / rolling_sum(pd.Series(TR), n), 
        name = name)
    df[name] = VI
    return df

#KST Oscillator
def KST(df, r1, r2, r3, r4, n1, n2, n3, n4):
    M = df['Close'].diff(r1 - 1)
    N = df['Close'].shift(r1 - 1)
    ROC1 = M / N
    M = df['Close'].diff(r2 - 1)
    N = df['Close'].shift(r2 - 1)
    ROC2 = M / N
    M = df['Close'].diff(r3 - 1)
    N = df['Close'].shift(r3 - 1)
    ROC3 = M / N
    M = df['Close'].diff(r4 - 1)
    N = df['Close'].shift(r4 - 1)
    ROC4 = M / N
    KST = pd.Series(rolling_sum(ROC1, n1) + rolling_sum(ROC2, n2) * 2 + rolling_sum(ROC3, n3) * 3 + rolling_sum(ROC4, n4) * 4, name = 'KST_' + str(r1) + '_' + str(r2) + '_' + str(r3) + '_' + str(r4) + '_' + str(n1) + '_' + str(n2) + '_' + str(n3) + '_' + str(n4))
    df['KST'] = KST
    return df

#Relative Strength Index
def RSI(df, n):
    name = 'RSI_' + str(n)
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= len(df) - 1:
        UpMove = get_value(df, i + 1, 'High') - get_value(df, i, 'High')
        DoMove = get_value(df, i, 'Low') - get_value(df, i + 1, 'Low')
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else: UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else: DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(ewma(UpI, span = n, min_periods = n - 1))
    NegDI = pd.Series(ewma(DoI, span = n, min_periods = n - 1))
    RSI = pd.Series(PosDI / (PosDI + NegDI), name = name)
    df[name] = RSI
    return df

#True Strength Index
def TSI(df, r, s):
    name = 'TSI_' + str(r) + '_' + str(s)
    M = pd.Series(df['Close'].diff(1))
    aM = abs(M)
    EMA1 = pd.Series(ewma(M, span = r, min_periods = r - 1))
    aEMA1 = pd.Series(ewma(aM, span = r, min_periods = r - 1))
    EMA2 = pd.Series(ewma(EMA1, span = s, min_periods = s - 1))
    aEMA2 = pd.Series(ewma(aEMA1, span = s, min_periods = s - 1))
    TSI = pd.Series(EMA2 / aEMA2, name = name)
    df[name] = TSI
    return df

#Accumulation/Distribution
def ACCDIST(df, n):
    name = 'Acc/Dist_ROC_' + str(n)
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    M = ad.diff(n - 1)
    N = ad.shift(n - 1)
    ROC = M / N
    AD = pd.Series(ROC, name = name)
    df[name] = AD
    return df

#Chaikin Oscillator
def Chaikin(df):
    ad = (2 * df['Close'] - df['High'] - df['Low']) / (df['High'] - df['Low']) * df['Volume']
    Chaikin = pd.Series(
        ewma(ad, span = 3, min_periods = 2) - ewma(ad, span = 10, min_periods = 9), 
        name = 'Chaikin')
    df['Chaikin'] = Chaikin
    return df

#Money Flow Index and Ratio
def MFI(df, n):
    name = 'MFI_' + str(n)
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    i = 0
    PosMF = [0]
    while i < len(df) - 1:
        if PP.iloc[i + 1] > PP.iloc[i]:
            PosMF.append(PP.iloc[i + 1] * get_value(df, i + 1, 'Volume'))
        else:
            PosMF.append(0)
        i = i + 1
    PosMF = pd.Series(PosMF)
    TotMF = PP * df['Volume']
    MFR = pd.Series(PosMF / TotMF)
    MFI = pd.Series(rolling_mean(MFR, n), name = name)
    df[name] = MFI
    return df

#On-balance Volume
def OBV(df, n):
    name = 'OBV_' + str(n)
    i = 0
    OBV = [0]
    while i < len(df) - 1:
        if get_value(df, i + 1, 'Close') - get_value(df, i, 'Close') > 0:
            OBV.append(get_value(df, i + 1, 'Volume'))
        if get_value(df, i + 1, 'Close') - get_value(df, i, 'Close') == 0:
            OBV.append(0)
        if get_value(df, i + 1, 'Close') - get_value(df, i, 'Close') < 0:
            OBV.append(-get_value(df, i + 1, 'Volume'))
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(rolling_mean(OBV, n), name = name)
    df[name] = OBV_ma
    return df

#Force Index
def FORCE(df, n):
    name = 'Force_' + str(n)
    F = pd.Series(df['Close'].diff(n) * df['Volume'].diff(n), name = name)
    df[name] = F
    return df

#Ease of Movement
def EOM(df, n):
    name = 'EoM_' + str(n)
    EoM = (df['High'].diff(1) + df['Low'].diff(1)) * (df['High'] - df['Low']) / (2 * df['Volume'])
    Eom_ma = pd.Series(rolling_mean(EoM, n), name = name)
    df[name] = Eom_ma
    return df

#Commodity Channel Index
def CCI(df, n):
    name = 'CCI_' + str(n)
    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series(
        (PP - rolling_mean(PP, n)) / rolling_std(PP, n), 
        name = name)
    df[name] = CCI
    return df

#Coppock Curve
def COPP(df, n):
    name = 'Copp_' + str(n)
    M = df['Close'].diff(int(n * 11 / 10) - 1)
    N = df['Close'].shift(int(n * 11 / 10) - 1)
    ROC1 = M / N
    M = df['Close'].diff(int(n * 14 / 10) - 1)
    N = df['Close'].shift(int(n * 14 / 10) - 1)
    ROC2 = M / N
    Copp = pd.Series(
        ewma(ROC1 + ROC2, span = n, min_periods = n), 
        name = name)
    df[name] = Copp
    return df

#Keltner Channel
def KELCH(df, n):
    name_M = 'KelChM_' + str(n)
    name_U = 'KelChU_' + str(n)
    name_D = 'KelChD_' + str(n)
    KelChM = pd.Series(
        rolling_mean((df['High'] + df['Low'] + df['Close']) / 3, n), 
        name = name_M)
    KelChU = pd.Series(
        rolling_mean((4 * df['High'] - 2 * df['Low'] + df['Close']) / 3, n), 
        name = name_U)
    KelChD = pd.Series(
        rolling_mean((-2 * df['High'] + 4 * df['Low'] + df['Close']) / 3, n), 
        name = name_D)
    df[name_M] = KelChM
    df[name_U] = KelChU
    df[name_D] = KelChD
    return df

#Ultimate Oscillator
def ULTOSC(df):
    i = 0
    TR_l = [0]
    BP_l = [0]
    while i < len(df) - 1:
        TR = max(get_value(df, i + 1, 'High'), get_value(df, i, 'Close')) - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        TR_l.append(TR)
        BP = get_value(df, i + 1, 'Close') - min(get_value(df, i + 1, 'Low'), get_value(df, i, 'Close'))
        BP_l.append(BP)
        i = i + 1
    UltO = pd.Series((4 * rolling_sum(pd.Series(BP_l), 7) / rolling_sum(pd.Series(TR_l), 7)) + (2 * rolling_sum(pd.Series(BP_l), 14) / rolling_sum(pd.Series(TR_l), 14)) + (rolling_sum(pd.Series(BP_l), 28) / rolling_sum(pd.Series(TR_l), 28)), name = 'Ultimate_Osc')
    df['UltO'] = UltO
    return df

#Donchian Channel
def DONCH(df, n):
    name = 'Donchian_' + str(n)
    i = 0
    DC_l = []
    while i < n - 1:
        DC_l.append(0)
        i = i + 1
    i = 0
    while i + n - 1 < len(df) - 1:
        DC = max(df['High'].iloc[i:i + n - 1]) - min(df['Low'].iloc[i:i + n - 1])
        DC_l.append(DC)
        i = i + 1
    DonCh = pd.Series(DC_l, name = name)
    DonCh = DonCh.shift(n - 1)
    df[name] = DonCh
    return df

#Standard Deviation
def STDDEV(df, n):
    name = 'STD_' + str(n)
    std_dev = pd.Series(rolling_std(df['Close'], n), name = name)
    df[name] = std_dev
    return df