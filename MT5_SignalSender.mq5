//+------------------------------------------------------------------+
//|                                              MT5_SignalSender.mq5 |
//|                        Copyright 2024, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Trading Bot System"
#property link      "https://www.example.com"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include <Json\JAson.mqh>

//--- Input parameters
input string   LinuxBotURL = "http://192.168.1.100:8080";  // Linux bot IP and port
input string   APIEndpoint = "/api/signals";                // Signal endpoint
input int      SignalInterval = 60;                         // Signal interval in seconds
input double   MinPriceChange = 0.0010;                    // Minimum price change to trigger signal
input bool     EnableSignals = true;                        // Enable/disable signal sending
input string   TradingSymbols = "EURUSD,GBPUSD,USDJPY";    // Symbols to monitor

//--- Global variables
CTrade trade;
datetime lastSignalTime[];
double lastPrice[];
string symbols[];
int symbolCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("MT5 Signal Sender initialized");
    
    // Parse trading symbols
    ParseSymbols();
    
    // Initialize arrays
    ArrayResize(lastSignalTime, symbolCount);
    ArrayResize(lastPrice, symbolCount);
    
    // Initialize last prices
    for(int i = 0; i < symbolCount; i++)
    {
        lastSignalTime[i] = 0;
        lastPrice[i] = SymbolInfoDouble(symbols[i], SYMBOL_BID);
    }
    
    Print("Monitoring ", symbolCount, " symbols for signals");
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("MT5 Signal Sender stopped. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    if(!EnableSignals) return;
    
    datetime currentTime = TimeCurrent();
    
    for(int i = 0; i < symbolCount; i++)
    {
        string symbol = symbols[i];
        
        // Check if enough time has passed since last signal
        if(currentTime - lastSignalTime[i] >= SignalInterval)
        {
            CheckAndSendSignal(symbol, i);
        }
    }
}

//+------------------------------------------------------------------+
//| Parse symbols from input string                                  |
//+------------------------------------------------------------------+
void ParseSymbols()
{
    string symbolString = TradingSymbols;
    symbolCount = 0;
    
    // Count symbols
    for(int i = 0; i < StringLen(symbolString); i++)
    {
        if(StringGetCharacter(symbolString, i) == ',')
            symbolCount++;
    }
    symbolCount++; // Add one for the last symbol
    
    ArrayResize(symbols, symbolCount);
    
    // Extract symbols
    int start = 0;
    int symbolIndex = 0;
    
    for(int i = 0; i <= StringLen(symbolString); i++)
    {
        if(i == StringLen(symbolString) || StringGetCharacter(symbolString, i) == ',')
        {
            symbols[symbolIndex] = StringSubstr(symbolString, start, i - start);
            StringTrimLeft(symbols[symbolIndex]);
            StringTrimRight(symbols[symbolIndex]);
            symbolIndex++;
            start = i + 1;
        }
    }
}

//+------------------------------------------------------------------+
//| Check market conditions and send signal if needed               |
//+------------------------------------------------------------------+
void CheckAndSendSignal(string symbol, int index)
{
    double currentPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
    double priceChange = MathAbs(currentPrice - lastPrice[index]);
    
    if(priceChange >= MinPriceChange)
    {
        // Gather market data
        MarketData marketData;
        GatherMarketData(symbol, marketData);
        
        // Create and send signal
        SendSignalToLinuxBot(symbol, marketData);
        
        // Update tracking variables
        lastSignalTime[index] = TimeCurrent();
        lastPrice[index] = currentPrice;
    }
}

//+------------------------------------------------------------------+
//| Market data structure                                            |
//+------------------------------------------------------------------+
struct MarketData
{
    string symbol;
    datetime timestamp;
    double bid;
    double ask;
    double spread;
    long volume;
    double high;
    double low;
    double open;
    double close;
    double rsi;
    double ma_fast;
    double ma_slow;
    double bollinger_upper;
    double bollinger_lower;
    double atr;
};

//+------------------------------------------------------------------+
//| Gather comprehensive market data                                 |
//+------------------------------------------------------------------+
void GatherMarketData(string symbol, MarketData &data)
{
    data.symbol = symbol;
    data.timestamp = TimeCurrent();
    data.bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    data.ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    data.spread = data.ask - data.bid;
    data.volume = SymbolInfoInteger(symbol, SYMBOL_VOLUME);
    
    // Get current bar data
    MqlRates rates[];
    if(CopyRates(symbol, PERIOD_M1, 0, 1, rates) > 0)
    {
        data.high = rates[0].high;
        data.low = rates[0].low;
        data.open = rates[0].open;
        data.close = rates[0].close;
    }
    
    // Calculate technical indicators
    CalculateIndicators(symbol, data);
}

//+------------------------------------------------------------------+
//| Calculate technical indicators                                   |
//+------------------------------------------------------------------+
void CalculateIndicators(string symbol, MarketData &data)
{
    // RSI (14 period)
    int rsiHandle = iRSI(symbol, PERIOD_M5, 14, PRICE_CLOSE);
    if(rsiHandle != INVALID_HANDLE)
    {
        double rsiBuffer[];
        if(CopyBuffer(rsiHandle, 0, 0, 1, rsiBuffer) > 0)
            data.rsi = rsiBuffer[0];
        IndicatorRelease(rsiHandle);
    }
    
    // Moving Averages
    int maFastHandle = iMA(symbol, PERIOD_M5, 10, 0, MODE_SMA, PRICE_CLOSE);
    int maSlowHandle = iMA(symbol, PERIOD_M5, 20, 0, MODE_SMA, PRICE_CLOSE);
    
    if(maFastHandle != INVALID_HANDLE)
    {
        double maFastBuffer[];
        if(CopyBuffer(maFastHandle, 0, 0, 1, maFastBuffer) > 0)
            data.ma_fast = maFastBuffer[0];
        IndicatorRelease(maFastHandle);
    }
    
    if(maSlowHandle != INVALID_HANDLE)
    {
        double maSlowBuffer[];
        if(CopyBuffer(maSlowHandle, 0, 0, 1, maSlowBuffer) > 0)
            data.ma_slow = maSlowBuffer[0];
        IndicatorRelease(maSlowHandle);
    }
    
    // Bollinger Bands
    int bbHandle = iBands(symbol, PERIOD_M5, 20, 0, 2.0, PRICE_CLOSE);
    if(bbHandle != INVALID_HANDLE)
    {
        double upperBuffer[], lowerBuffer[];
        if(CopyBuffer(bbHandle, 1, 0, 1, upperBuffer) > 0)
            data.bollinger_upper = upperBuffer[0];
        if(CopyBuffer(bbHandle, 2, 0, 1, lowerBuffer) > 0)
            data.bollinger_lower = lowerBuffer[0];
        IndicatorRelease(bbHandle);
    }
    
    // ATR (14 period)
    int atrHandle = iATR(symbol, PERIOD_M5, 14);
    if(atrHandle != INVALID_HANDLE)
    {
        double atrBuffer[];
        if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) > 0)
            data.atr = atrBuffer[0];
        IndicatorRelease(atrHandle);
    }
}

//+------------------------------------------------------------------+
//| Send signal to Linux bot via HTTP request                       |
//+------------------------------------------------------------------+
void SendSignalToLinuxBot(string symbol, MarketData &data)
{
    string url = LinuxBotURL + APIEndpoint;
    string jsonPayload = CreateJSONPayload(data);
    
    char post[], result[];
    string headers;
    
    // Convert string to char array
    StringToCharArray(jsonPayload, post, 0, StringLen(jsonPayload));
    
    // Set headers
    headers = "Content-Type: application/json\r\n";
    
    // Send HTTP POST request
    int timeout = 5000; // 5 seconds timeout
    int res = WebRequest("POST", url, headers, timeout, post, result, headers);
    
    if(res == 200)
    {
        Print("Signal sent successfully for ", symbol);
        Print("Response: ", CharArrayToString(result));
    }
    else
    {
        Print("Failed to send signal for ", symbol, ". HTTP code: ", res);
        Print("Error: ", CharArrayToString(result));
    }
}

//+------------------------------------------------------------------+
//| Create JSON payload from market data                            |
//+------------------------------------------------------------------+
string CreateJSONPayload(MarketData &data)
{
    string json = "{";
    json += "\"signal_type\":\"market_data\",";
    json += "\"symbol\":\"" + data.symbol + "\",";
    json += "\"timestamp\":\"" + TimeToString(data.timestamp, TIME_DATE|TIME_SECONDS) + "\",";
    json += "\"bid\":" + DoubleToString(data.bid, 5) + ",";
    json += "\"ask\":" + DoubleToString(data.ask, 5) + ",";
    json += "\"spread\":" + DoubleToString(data.spread, 5) + ",";
    json += "\"volume\":" + IntegerToString(data.volume) + ",";
    json += "\"ohlc\":{";
    json += "\"open\":" + DoubleToString(data.open, 5) + ",";
    json += "\"high\":" + DoubleToString(data.high, 5) + ",";
    json += "\"low\":" + DoubleToString(data.low, 5) + ",";
    json += "\"close\":" + DoubleToString(data.close, 5);
    json += "},";
    json += "\"indicators\":{";
    json += "\"rsi\":" + DoubleToString(data.rsi, 2) + ",";
    json += "\"ma_fast\":" + DoubleToString(data.ma_fast, 5) + ",";
    json += "\"ma_slow\":" + DoubleToString(data.ma_slow, 5) + ",";
    json += "\"bollinger_upper\":" + DoubleToString(data.bollinger_upper, 5) + ",";
    json += "\"bollinger_lower\":" + DoubleToString(data.bollinger_lower, 5) + ",";
    json += "\"atr\":" + DoubleToString(data.atr, 5);
    json += "},";
    json += "\"source\":\"MT5_Windows_VM\",";
    json += "\"version\":\"1.0\"";
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Handle trade events                                              |
//+------------------------------------------------------------------+
void OnTrade()
{
    // Send trade execution results to Linux bot
    SendTradeExecutionUpdate();
}

//+------------------------------------------------------------------+
//| Send trade execution updates                                     |
//+------------------------------------------------------------------+
void SendTradeExecutionUpdate()
{
    // Get latest trade from history
    HistorySelect(TimeCurrent() - 3600, TimeCurrent()); // Last hour
    int total = HistoryDealsTotal();
    
    if(total > 0)
    {
        ulong ticket = HistoryDealGetTicket(total - 1);
        if(ticket > 0)
        {
            string symbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
            double volume = HistoryDealGetDouble(ticket, DEAL_VOLUME);
            double price = HistoryDealGetDouble(ticket, DEAL_PRICE);
            ENUM_DEAL_TYPE dealType = (ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket, DEAL_TYPE);
            datetime dealTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
            
            string json = "{";
            json += "\"signal_type\":\"trade_execution\",";
            json += "\"ticket\":" + IntegerToString(ticket) + ",";
            json += "\"symbol\":\"" + symbol + "\",";
            json += "\"volume\":" + DoubleToString(volume, 2) + ",";
            json += "\"price\":" + DoubleToString(price, 5) + ",";
            json += "\"type\":\"" + EnumToString(dealType) + "\",";
            json += "\"timestamp\":\"" + TimeToString(dealTime, TIME_DATE|TIME_SECONDS) + "\",";
            json += "\"source\":\"MT5_Windows_VM\"";
            json += "}";
            
            string url = LinuxBotURL + "/api/trade-execution";
            SendHTTPRequest(url, json);
        }
    }
}

//+------------------------------------------------------------------+
//| Generic HTTP request sender                                      |
//+------------------------------------------------------------------+
void SendHTTPRequest(string url, string jsonPayload)
{
    char post[], result[];
    string headers = "Content-Type: application/json\r\n";
    
    StringToCharArray(jsonPayload, post, 0, StringLen(jsonPayload));
    
    int res = WebRequest("POST", url, headers, 5000, post, result, headers);
    
    if(res == 200)
    {
        Print("HTTP request sent successfully to: ", url);
    }
    else
    {
        Print("HTTP request failed. URL: ", url, ", Code: ", res);
    }
}