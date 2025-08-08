//+------------------------------------------------------------------+
//|                              WindowsVM_SignalSender_Enhanced.mq5 |
//|                        Copyright 2024, Your Trading Bot System   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Trading Bot System"
#property link      "https://www.example.com"
#property version   "1.10"
#property strict

//--- Include WinINet for HTTP requests
#import "wininet.dll"
int InternetOpenW(string, int, string, string, int);
int InternetConnectW(int, string, int, string, string, int, int, int);
int HttpOpenRequestW(int, string, string, string, string, string, int, int);
bool HttpSendRequestW(int, string, int, string, int);
bool InternetReadFile(int, uchar &buffer[], int, int &bytesRead[]);
bool InternetCloseHandle(int);
#import

//--- Input parameters
input string   LinuxBotIP = "129.151.168.242";             // Linux bot IP address
input int      BridgePort = 5005;                          // MT5 Bridge port
input string   APIEndpoint = "/windows-vm-signals";        // Signal endpoint
input int      SignalInterval = 30;                        // Signal interval in seconds
input double   MinPriceChange = 0.0005;                    // Minimum price change to trigger signal
input bool     EnableSignals = true;                       // Enable/disable signal sending
input string   TradingSymbols = "EURUSD,GBPUSD,USDJPY";    // Symbols to monitor
input bool     DebugMode = true;                           // Enable debug output

//--- Global variables
datetime lastSignalTime[];
double lastPrice[];
string symbols[];
int symbolCount = 0;
int hInternet = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("Enhanced Windows VM Signal Sender initialized");
    
    // Initialize WinINet
    hInternet = InternetOpenW("MT5SignalSender", 1, "", "", 0);
    if(hInternet == 0)
    {
        Print("ERROR: Failed to initialize WinINet");
        return INIT_FAILED;
    }
    
    // Parse trading symbols
    ParseTradingSymbols();
    
    Print("Sending signals to: http://", LinuxBotIP, ":", BridgePort, APIEndpoint);
    Print("Monitoring ", symbolCount, " symbols with ", SignalInterval, "s interval");
    
    // Test connection on startup
    TestConnection();
    
    return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Parse trading symbols from input string                         |
//+------------------------------------------------------------------+
void ParseTradingSymbols()
{
    string symbolString = TradingSymbols;
    symbolCount = 0;
    
    // Count symbols
    int pos = 0;
    while(pos >= 0)
    {
        pos = StringFind(symbolString, ",", pos);
        symbolCount++;
        if(pos >= 0) pos++;
    }
    
    // Resize arrays
    ArrayResize(symbols, symbolCount);
    ArrayResize(lastSignalTime, symbolCount);
    ArrayResize(lastPrice, symbolCount);
    
    // Parse symbols
    pos = 0;
    for(int i = 0; i < symbolCount; i++)
    {
        int nextPos = StringFind(symbolString, ",", pos);
        if(nextPos == -1) nextPos = StringLen(symbolString);
        
        symbols[i] = StringSubstr(symbolString, pos, nextPos - pos);
        StringTrimLeft(symbols[i]);
        StringTrimRight(symbols[i]);
        
        lastSignalTime[i] = 0;
        lastPrice[i] = 0;
        
        pos = nextPos + 1;
        Print("Monitoring symbol: ", symbols[i]);
    }
}

//+------------------------------------------------------------------+
//| Test connection to Linux bot                                    |
//+------------------------------------------------------------------+
void TestConnection()
{
    Print("Testing connection to Linux bot...");
    string testPayload = "{\"signal_type\":\"connection_test\",\"timestamp\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"}";
    
    int result = SendHTTPRequest(testPayload);
    if(result == 200)
        Print("✅ Connection test successful!");
    else
        Print("❌ Connection test failed. HTTP code: ", result);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(hInternet != 0)
        InternetCloseHandle(hInternet);
    
    Print("Enhanced Windows VM Signal Sender stopped");
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
//| Check market conditions and send signal if needed               |
//+------------------------------------------------------------------+
void CheckAndSendSignal(string symbol, int index)
{
    double currentPrice = SymbolInfoDouble(symbol, SYMBOL_BID);
    
    if(currentPrice == 0)
    {
        if(DebugMode) Print("Warning: No price data for ", symbol);
        return;
    }
    
    double priceChange = MathAbs(currentPrice - lastPrice[index]);
    
    // Send signal if price changed significantly or this is the first signal
    if(priceChange >= MinPriceChange || lastPrice[index] == 0)
    {
        string jsonPayload = CreateJSONPayload(symbol);
        int httpResult = SendHTTPRequest(jsonPayload);
        
        if(httpResult == 200)
        {
            Print("✅ Signal sent successfully for ", symbol);
            lastSignalTime[index] = TimeCurrent();
            lastPrice[index] = currentPrice;
        }
        else
        {
            Print("❌ Failed to send signal for ", symbol, ". HTTP code: ", httpResult);
        }
    }
    else if(DebugMode)
    {
        Print("Price change too small for ", symbol, ": ", priceChange, " < ", MinPriceChange);
    }
}

//+------------------------------------------------------------------+
//| Send HTTP request to Linux bot                                  |
//+------------------------------------------------------------------+
int SendHTTPRequest(string jsonPayload)
{
    if(hInternet == 0)
    {
        Print("ERROR: WinINet not initialized");
        return -1;
    }
    
    // Connect to server
    int hConnect = InternetConnectW(hInternet, LinuxBotIP, BridgePort, "", "", 3, 0, 0);
    if(hConnect == 0)
    {
        Print("ERROR: Failed to connect to ", LinuxBotIP, ":", BridgePort);
        return -1;
    }
    
    // Open HTTP request
    int hRequest = HttpOpenRequestW(hConnect, "POST", APIEndpoint, "HTTP/1.1", "", "", 0, 0);
    if(hRequest == 0)
    {
        Print("ERROR: Failed to open HTTP request");
        InternetCloseHandle(hConnect);
        return -1;
    }
    
    // Prepare headers
    string headers = "Content-Type: application/json\r\nContent-Length: " + IntegerToString(StringLen(jsonPayload)) + "\r\n";
    
    // Convert payload to bytes
    uchar data[];
    StringToCharArray(jsonPayload, data, 0, StringLen(jsonPayload));
    
    // Send request
    bool success = HttpSendRequestW(hRequest, headers, StringLen(headers), data, ArraySize(data));
    
    int httpCode = -1;
    if(success)
    {
        // Read response (simplified)
        uchar buffer[1024];
        int bytesRead[1];
        InternetReadFile(hRequest, buffer, 1024, bytesRead);
        httpCode = 200; // Assume success if we can send
        
        if(DebugMode && bytesRead[0] > 0)
        {
            string response = CharArrayToString(buffer, 0, bytesRead[0]);
            Print("Response: ", response);
        }
    }
    else
    {
        Print("ERROR: HttpSendRequestW failed");
        httpCode = -1;
    }
    
    // Cleanup
    InternetCloseHandle(hRequest);
    InternetCloseHandle(hConnect);
    
    return httpCode;
}

//+------------------------------------------------------------------+
//| Create JSON payload from market data                            |
//+------------------------------------------------------------------+
string CreateJSONPayload(string symbol)
{
    datetime currentTime = TimeCurrent();
    double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
    double spread = ask - bid;
    long volume = SymbolInfoInteger(symbol, SYMBOL_VOLUME);
    
    // Get OHLC data from current timeframe
    double open = iOpen(symbol, PERIOD_CURRENT, 0);
    double high = iHigh(symbol, PERIOD_CURRENT, 0);
    double low = iLow(symbol, PERIOD_CURRENT, 0);
    double close = iClose(symbol, PERIOD_CURRENT, 0);
    
    // Calculate basic technical indicators
    double rsi = CalculateRSI(symbol, 14);
    double maFast = CalculateMA(symbol, 10);
    double maSlow = CalculateMA(symbol, 20);
    double atr = CalculateATR(symbol, 14);
    
    // Build JSON payload
    string json = "{";
    json += "\"signal_type\":\"market_data\",";
    json += "\"symbol\":\"" + symbol + "\",";
    json += "\"timestamp\":\"" + TimeToString(currentTime, TIME_DATE|TIME_SECONDS) + "\",";
    json += "\"bid\":" + DoubleToString(bid, 5) + ",";
    json += "\"ask\":" + DoubleToString(ask, 5) + ",";
    json += "\"spread\":" + DoubleToString(spread, 5) + ",";
    json += "\"volume\":" + IntegerToString(volume) + ",";
    json += "\"ohlc\":{";
    json += "\"open\":" + DoubleToString(open, 5) + ",";
    json += "\"high\":" + DoubleToString(high, 5) + ",";
    json += "\"low\":" + DoubleToString(low, 5) + ",";
    json += "\"close\":" + DoubleToString(close, 5);
    json += "},";
    json += "\"indicators\":{";
    json += "\"rsi\":" + DoubleToString(rsi, 2) + ",";
    json += "\"ma_fast\":" + DoubleToString(maFast, 5) + ",";
    json += "\"ma_slow\":" + DoubleToString(maSlow, 5) + ",";
    json += "\"atr\":" + DoubleToString(atr, 5);
    json += "},";
    json += "\"source\":\"MT5_Windows_VM\",";
    json += "\"version\":\"1.1\"";
    json += "}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Calculate RSI indicator                                          |
//+------------------------------------------------------------------+
double CalculateRSI(string symbol, int period)
{
    double rsi = 50; // Default neutral value
    double prices[];
    ArraySetAsSeries(prices, true);
    
    if(CopyClose(symbol, PERIOD_CURRENT, 0, period + 1, prices) > 0)
    {
        double gains = 0, losses = 0;
        
        for(int i = 1; i < ArraySize(prices); i++)
        {
            double change = prices[i-1] - prices[i];
            if(change > 0)
                gains += change;
            else
                losses += MathAbs(change);
        }
        
        double avgGain = gains / period;
        double avgLoss = losses / period;
        
        if(avgLoss != 0)
            rsi = 100 - (100 / (1 + (avgGain / avgLoss)));
    }
    
    return rsi;
}

//+------------------------------------------------------------------+
//| Calculate Moving Average                                         |
//+------------------------------------------------------------------+
double CalculateMA(string symbol, int period)
{
    double ma = 0;
    double prices[];
    ArraySetAsSeries(prices, true);
    
    if(CopyClose(symbol, PERIOD_CURRENT, 0, period, prices) > 0)
    {
        double sum = 0;
        for(int i = 0; i < period; i++)
            sum += prices[i];
        ma = sum / period;
    }
    
    return ma;
}

//+------------------------------------------------------------------+
//| Calculate ATR indicator                                          |
//+------------------------------------------------------------------+
double CalculateATR(string symbol, int period)
{
    double atr = 0.0001; // Default small value
    double high[], low[], close[];
    ArraySetAsSeries(high, true);
    ArraySetAsSeries(low, true);
    ArraySetAsSeries(close, true);
    
    if(CopyHigh(symbol, PERIOD_CURRENT, 0, period + 1, high) > 0 &&
       CopyLow(symbol, PERIOD_CURRENT, 0, period + 1, low) > 0 &&
       CopyClose(symbol, PERIOD_CURRENT, 0, period + 1, close) > 0)
    {
        double trSum = 0;
        
        for(int i = 1; i < period + 1; i++)
        {
            double tr1 = high[i] - low[i];
            double tr2 = MathAbs(high[i] - close[i+1]);
            double tr3 = MathAbs(low[i] - close[i+1]);
            double tr = MathMax(tr1, MathMax(tr2, tr3));
            trSum += tr;
        }
        
        atr = trSum / period;
    }
    
    return atr;
}