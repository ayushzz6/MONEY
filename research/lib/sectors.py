"""NSE sectoral indices for the strategy library: Angel One tokens, and each NIFTY 500 industry's index.

Banks are split from the rest of financial services (PSU banks / private banks have their own indices).
"""
SECTOR_TOKENS = {
    "SEC_AUTO": "99926029", "SEC_IT": "99926008", "SEC_PHARMA": "99926023", "SEC_FMCG": "99926021",
    "SEC_METAL": "99926030", "SEC_REALTY": "99926018", "SEC_ENERGY": "99926020", "SEC_FINSRV": "99926037",
    "SEC_PSUBANK": "99926025", "SEC_PVTBANK": "99926047", "SEC_MEDIA": "99926031", "SEC_INFRA": "99926019",
    "SEC_COMMOD": "99926035", "SEC_CONSUMP": "99926036", "SEC_SERVICES": "99926026", "SEC_PSE": "99926024",
    "SEC_MIDCAP100": "99926011", "SEC_SMLCAP100": "99926032", "SEC_NIFTY500": "99926004",
}
INDUSTRY_INDEX = {
    "Automobile and Auto Components": "SEC_AUTO", "Information Technology": "SEC_IT", "Healthcare": "SEC_PHARMA",
    "Fast Moving Consumer Goods": "SEC_FMCG", "Metals & Mining": "SEC_METAL", "Realty": "SEC_REALTY",
    "Oil Gas & Consumable Fuels": "SEC_ENERGY", "Power": "SEC_ENERGY", "Financial Services": "SEC_FINSRV",
    "Media Entertainment & Publication": "SEC_MEDIA", "Capital Goods": "SEC_INFRA", "Construction": "SEC_INFRA",
    "Telecommunication": "SEC_INFRA", "Construction Materials": "SEC_COMMOD", "Chemicals": "SEC_COMMOD",
    "Consumer Durables": "SEC_CONSUMP", "Consumer Services": "SEC_CONSUMP", "Textiles": "SEC_CONSUMP",
    "Services": "SEC_SERVICES", "Diversified": "SEC_NIFTY500",
}
PSU_BANKS = {"SBIN", "BANKBARODA", "PNB", "CANBK", "UNIONBANK", "INDIANB", "BANKINDIA", "IOB", "UCOBANK",
             "CENTRALBK", "MAHABANK", "PSB"}
PVT_BANKS = {"HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "IDFCFIRSTB", "FEDERALBNK",
             "BANDHANBNK", "RBLBANK", "YESBANK", "AUBANK", "CUB", "KARURVYSYA", "J&KBANK", "IDBI"}


def sector_index(symbol: str, industry: str) -> str:
    if symbol in PSU_BANKS:
        return "SEC_PSUBANK"
    if symbol in PVT_BANKS:
        return "SEC_PVTBANK"
    return INDUSTRY_INDEX.get(industry, "SEC_NIFTY500")
