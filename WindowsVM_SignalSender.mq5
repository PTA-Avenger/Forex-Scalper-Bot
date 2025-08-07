//+------------------------------------------------------------------+
//|                                        WindowsVM_SignalSender.mq5 |
//|                        Copyright 2024, Your Trading Bot System   |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Trading Bot System"
#property link      "https://www.example.com"
#property version   "1.00"
#property strict

//--- Input parameters
input string   LinuxBotIP = "192.168.1.100";              // Linux bot IP address
input int      BridgePort = 5005;                          // MT5 Bridge port
input string   APIEndpoint = "/windows-vm-signals";        // Signal endpoint
input int      SignalInterval = 30;                        // Signal interval in seconds
input double   MinPriceChange = 0.0005;                    // Minimum price change to trigger signal
input bool     EnableSignals = true;                       // Enable/disable signal sending
input string   TradingSymbols = "EURUSD,GBPUSD,USDJPY";    // Symbols to monitor

//--- Global variables
datetime lastSignalTime[];
double lastPrice[];
string symbols[];
int symbolCount = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("Windows VM Signal Sender initialized");
    
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
    Print("Sending signals to: http://", LinuxBotIP, ":", BridgePort, APIEndpoint);
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    Print("Windows VM Signal Sender stopped. Reason: ", reason);
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
        // Create and send signal
        SendSignalToLinuxBot(symbol);
        
        // Update tracking variables
        lastSignalTime[index] = TimeCurrent();
        lastPrice[index] = currentPrice;
    }
}

//+------------------------------------------------------------------+
//| Send signal to Linux bot via HTTP request                       |
//+------------------------------------------------------------------+
void SendSignalToLinuxBot(string symbol)
{
    string url = "http://" + LinuxBotIP + ":" + IntegerToString(BridgePort) + APIEndpoint;
    string jsonPayload = CreateJSONPayload(symbol);
    
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
        string responseStr = CharArrayToString(result);
        Print("Response: ", responseStr);
    }
    else
    {
        Print("Failed to send signal for ", symbol, ". HTTP code: ", res);
        if(ArraySize(result) > 0)
        {
            Print("Error: ", CharArrayToString(result));
        }
    }
}

//+------------------------------------------------------------------+
//| Create JSON payload from market data                            |
//+------------------------------------------------------------------+
string CreateJSONPayload(string symbol)
{
    // Get current market data
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double spread = ask - bid;
    long volume = SymbolInfoInteger(symbol, SYMBOL_VOLUME);
    
    // Get current bar data
    MqlRates rates[];
    double open_price = bid, high = bid, low = bid, close_price = bid;
    
    if(CopyRates(symbol, PERIOD_M1, 0, 1, rates) > 0)
    {
        open_price = rates[0].open;
        high = rates[0].high;
        low = rates[0].low;
        close_price = rates[0].close;
    }
    
    // Calculate basic technical indicators
    double rsi = CalculateRSI(symbol, 14);
    double ma_fast = CalculateMA(symbol, 10);
    double ma_slow = CalculateMA(symbol, 20);
    double atr = CalculateATR(symbol, 14);
    
    // Create JSON payload
    string json = "{";
    json += "\"signal_type\":\"market_data\",";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",";
    json += "\"bid\":" + DoubleToString(bid, 5) + ",";
    json += "\"ask\":" + DoubleToString(ask, 5) + ",";
    json += "\"spread\":" + DoubleToString(spread, 5) + ",";
    json += "\"volume\":" + IntegerToString(volume) + ",";
    json += "\"ohlc\":{";
    json += "\"open\":" + DoubleToString(open_price, 5) + ",";
    json += "\"high\":" + DoubleToString(high, 5) + ",";
    json += "\"low\":" + DoubleToString(low, 5) + ",";
    json += "\"close\":" + DoubleToString(close_price, 5);
    json += "},";
    json += "\"indicators\":{";
    json += "\"rsi\":" + DoubleToString(rsi, 2) + ",";
    json += "\"ma_fast\":" + DoubleToString(ma_fast, 5) + ",";
    json += "\"ma_slow\":" + DoubleToString(ma_slow, 5) + ",";
    json += "\"atr\":" + DoubleToString(atr, 5);
    json += "},";
    json += "\"source\":\"MT5_Windows_VM\",";
    json += "\"version\":\"1.0\"";
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Calculate RSI indicator                                          |
//+------------------------------------------------------------------+
double CalculateRSI(string symbol, int period)
{
    int rsiHandle = iRSI(symbol, PERIOD_M5, period, PRICE_CLOSE);
    if(rsiHandle == INVALID_HANDLE)
        return 50.0; // Default neutral value
    
    double rsiBuffer[];
    if(CopyBuffer(rsiHandle, 0, 0, 1, rsiBuffer) > 0)
    {
        IndicatorRelease(rsiHandle);
        return rsiBuffer[0];
    }
    
    IndicatorRelease(rsiHandle);
    return 50.0;
}

//+------------------------------------------------------------------+
//| Calculate Moving Average                                         |
//+------------------------------------------------------------------+
double CalculateMA(string symbol, int period)
{
    int maHandle = iMA(symbol, PERIOD_M5, period, 0, MODE_SMA, PRICE_CLOSE);
    if(maHandle == INVALID_HANDLE)
        return SymbolInfoDouble(symbol, SYMBOL_BID); // Default to current price
    
    double maBuffer[];
    if(CopyBuffer(maHandle, 0, 0, 1, maBuffer) > 0)
    {
        IndicatorRelease(maHandle);
        return maBuffer[0];
    }
    
    IndicatorRelease(maHandle);
    return SymbolInfoDouble(symbol, SYMBOL_BID);
}

//+------------------------------------------------------------------+
//| Calculate ATR indicator                                          |
//+------------------------------------------------------------------+
double CalculateATR(string symbol, int period)
{
    int atrHandle = iATR(symbol, PERIOD_M5, period);
    if(atrHandle == INVALID_HANDLE)
        return 0.0001; // Default small ATR value
    
    double atrBuffer[];
    if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) > 0)
    {
        IndicatorRelease(atrHandle);
        return atrBuffer[0];
    }
    
    IndicatorRelease(atrHandle);
    return 0.0001;
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
            
            string url = "http://" + LinuxBotIP + ":" + IntegerToString(BridgePort) + "/trade-execution";
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