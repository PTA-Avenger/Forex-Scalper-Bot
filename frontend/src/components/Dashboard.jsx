import React, { useEffect, useState, useCallback } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  Snackbar,
  CircularProgress,
  Fab,
  Tooltip
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Settings as SettingsIcon
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';

// Component imports
import TradingChart from './trading/TradingChart';
import PositionTable from './positions/PositionTable';
import RiskMetrics from './risk/RiskMetrics';
import QuickStats from './dashboard/QuickStats';
import SystemStatus from './dashboard/SystemStatus';
import AlertPanel from './alerts/AlertPanel';
import StrategyConfig from './strategies/StrategyConfig';
import PerformanceChart from './analytics/PerformanceChart';
import AIInsights from './analytics/AIInsights';

// Redux actions
import {
  fetchDashboardData,
  updateTick,
  updatePosition,
  updateOrder,
  addAlert,
  setEngineStatus,
  startEngine,
  stopEngine
} from '../store/slices/tradingSlice';

// Hooks
import { useWebSocket } from '../hooks/useWebSocket';
import { useRealTimeData } from '../hooks/useRealTimeData';

// Styled components
const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  position: 'relative',
  '&:hover': {
    boxShadow: theme.shadows[4],
  },
}));

const FloatingActionButtons = styled(Box)(({ theme }) => ({
  position: 'fixed',
  bottom: theme.spacing(2),
  right: theme.spacing(2),
  display: 'flex',
  flexDirection: 'column',
  gap: theme.spacing(1),
  zIndex: 1000,
}));

const LoadingOverlay = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  backgroundColor: 'rgba(255, 255, 255, 0.8)',
  zIndex: 10,
}));

const Dashboard = () => {
  const dispatch = useDispatch();
  
  // Redux state
  const {
    isConnected,
    engineStatus,
    accountInfo,
    positions,
    orders,
    systemStatus,
    alerts,
    isLoading,
    error
  } = useSelector(state => state.trading);
  
  const { riskMetrics } = useSelector(state => state.risk);
  const { strategies } = useSelector(state => state.strategies);
  
  // Local state
  const [selectedSymbol, setSelectedSymbol] = useState('EUR/USD');
  const [refreshing, setRefreshing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState('info');
  
  // WebSocket connection for real-time updates
  const { lastMessage, connectionStatus, sendMessage } = useWebSocket(
    process.env.REACT_APP_WS_URL || 'ws://localhost:8080/ws'
  );
  
  // Real-time data hook
  const { marketData, latency } = useRealTimeData(selectedSymbol);
  
  // Initialize dashboard
  useEffect(() => {
    dispatch(fetchDashboardData());
  }, [dispatch]);
  
  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      try {
        const data = JSON.parse(lastMessage.data);
        handleWebSocketMessage(data);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    }
  }, [lastMessage]);
  
  // Handle WebSocket connection status
  useEffect(() => {
    dispatch(setEngineStatus({
      connected: connectionStatus === 'Connected',
      status: connectionStatus
    }));
  }, [connectionStatus, dispatch]);
  
  const handleWebSocketMessage = useCallback((data) => {
    switch (data.type) {
      case 'tick':
        dispatch(updateTick(data.payload));
        break;
        
      case 'position_update':
        dispatch(updatePosition(data.payload));
        break;
        
      case 'order_update':
        dispatch(updateOrder(data.payload));
        break;
        
      case 'alert':
        dispatch(addAlert(data.payload));
        showSnackbar(data.payload.message, data.payload.severity || 'info');
        break;
        
      case 'system_status':
        dispatch(setEngineStatus(data.payload));
        break;
        
      case 'error':
        showSnackbar(data.payload.message, 'error');
        break;
        
      default:
        console.log('Unknown message type:', data.type);
    }
  }, [dispatch]);
  
  const showSnackbar = (message, severity = 'info') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };
  
  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await dispatch(fetchDashboardData()).unwrap();
      showSnackbar('Dashboard refreshed successfully', 'success');
    } catch (error) {
      showSnackbar('Failed to refresh dashboard', 'error');
    } finally {
      setRefreshing(false);
    }
  };
  
  const handleStartEngine = async () => {
    try {
      await dispatch(startEngine()).unwrap();
      showSnackbar('Trading engine started', 'success');
    } catch (error) {
      showSnackbar('Failed to start engine', 'error');
    }
  };
  
  const handleStopEngine = async () => {
    try {
      await dispatch(stopEngine()).unwrap();
      showSnackbar('Trading engine stopped', 'warning');
    } catch (error) {
      showSnackbar('Failed to stop engine', 'error');
    }
  };
  
  const handleSymbolChange = (symbol) => {
    setSelectedSymbol(symbol);
  };
  
  const handleCloseSnackbar = (event, reason) => {
    if (reason === 'clickaway') {
      return;
    }
    setSnackbarOpen(false);
  };
  
  // Calculate grid layout based on screen size
  const getGridLayout = () => {
    return {
      systemStatus: { xs: 12, md: 12 },
      quickStats: { xs: 12, md: 8 },
      alerts: { xs: 12, md: 4 },
      chart: { xs: 12, lg: 8 },
      risk: { xs: 12, lg: 4 },
      positions: { xs: 12, md: 8 },
      aiInsights: { xs: 12, md: 4 },
      performance: { xs: 12, md: 6 },
      strategies: { xs: 12, md: 6 }
    };
  };
  
  const gridLayout = getGridLayout();
  
  return (
    <Box sx={{ flexGrow: 1, p: 3, pb: 10 }}>
      {/* Header */}
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Forex Scalping Bot Dashboard
        </Typography>
        
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" color="textSecondary">
            Latency: {latency}ms
          </Typography>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              backgroundColor: isConnected ? 'success.main' : 'error.main',
              animation: isConnected ? 'pulse 2s infinite' : 'none',
            }}
          />
        </Box>
      </Box>
      
      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => dispatch(clearError())}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        {/* System Status */}
        <Grid item {...gridLayout.systemStatus}>
          <StyledPaper>
            {isLoading && (
              <LoadingOverlay>
                <CircularProgress />
              </LoadingOverlay>
            )}
            <SystemStatus
              isConnected={isConnected}
              connectionStatus={connectionStatus}
              systemStatus={systemStatus}
              engineStatus={engineStatus}
            />
          </StyledPaper>
        </Grid>
        
        {/* Quick Stats */}
        <Grid item {...gridLayout.quickStats}>
          <StyledPaper>
            <QuickStats
              accountInfo={accountInfo}
              positions={positions}
              orders={orders}
              marketData={marketData}
            />
          </StyledPaper>
        </Grid>
        
        {/* Alerts Panel */}
        <Grid item {...gridLayout.alerts}>
          <StyledPaper>
            <AlertPanel alerts={alerts} />
          </StyledPaper>
        </Grid>
        
        {/* Trading Chart */}
        <Grid item {...gridLayout.chart}>
          <StyledPaper sx={{ height: 600 }}>
            <TradingChart
              symbol={selectedSymbol}
              onSymbolChange={handleSymbolChange}
              marketData={marketData}
              positions={positions.filter(p => p.symbol === selectedSymbol)}
              orders={orders.filter(o => o.symbol === selectedSymbol)}
            />
          </StyledPaper>
        </Grid>
        
        {/* Risk Metrics */}
        <Grid item {...gridLayout.risk}>
          <StyledPaper sx={{ height: 600 }}>
            <RiskMetrics
              riskMetrics={riskMetrics}
              accountInfo={accountInfo}
              positions={positions}
            />
          </StyledPaper>
        </Grid>
        
        {/* Positions Table */}
        <Grid item {...gridLayout.positions}>
          <StyledPaper>
            <PositionTable
              positions={positions}
              orders={orders}
              onClosePosition={(id) => dispatch(closePosition(id))}
              onCancelOrder={(id) => dispatch(cancelOrder(id))}
            />
          </StyledPaper>
        </Grid>
        
        {/* AI Insights */}
        <Grid item {...gridLayout.aiInsights}>
          <StyledPaper>
            <AIInsights
              symbol={selectedSymbol}
              marketData={marketData}
            />
          </StyledPaper>
        </Grid>
        
        {/* Performance Chart */}
        <Grid item {...gridLayout.performance}>
          <StyledPaper sx={{ height: 400 }}>
            <PerformanceChart
              accountInfo={accountInfo}
              timeRange="1D"
            />
          </StyledPaper>
        </Grid>
        
        {/* Strategy Configuration */}
        <Grid item {...gridLayout.strategies}>
          <StyledPaper sx={{ height: 400 }}>
            <StrategyConfig
              strategies={strategies}
              onUpdateStrategy={(name, params) => dispatch(updateStrategyParameters({ name, params }))}
              onToggleStrategy={(name, enabled) => dispatch(toggleStrategy({ name, enabled }))}
            />
          </StyledPaper>
        </Grid>
      </Grid>
      
      {/* Floating Action Buttons */}
      <FloatingActionButtons>
        <Tooltip title="Refresh Dashboard" placement="left">
          <Fab
            color="primary"
            size="medium"
            onClick={handleRefresh}
            disabled={refreshing}
          >
            {refreshing ? <CircularProgress size={24} /> : <RefreshIcon />}
          </Fab>
        </Tooltip>
        
        <Tooltip title={engineStatus?.running ? "Stop Engine" : "Start Engine"} placement="left">
          <Fab
            color={engineStatus?.running ? "error" : "success"}
            size="medium"
            onClick={engineStatus?.running ? handleStopEngine : handleStartEngine}
          >
            {engineStatus?.running ? <StopIcon /> : <PlayIcon />}
          </Fab>
        </Tooltip>
        
        <Tooltip title="Settings" placement="left">
          <Fab
            color="default"
            size="medium"
            onClick={() => setShowSettings(true)}
          >
            <SettingsIcon />
          </Fab>
        </Tooltip>
      </FloatingActionButtons>
      
      {/* Snackbar for notifications */}
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
      >
        <Alert
          onClose={handleCloseSnackbar}
          severity={snackbarSeverity}
          sx={{ width: '100%' }}
        >
          {snackbarMessage}
        </Alert>
      </Snackbar>
      
      {/* Settings Dialog */}
      {showSettings && (
        <SettingsDialog
          open={showSettings}
          onClose={() => setShowSettings(false)}
        />
      )}
    </Box>
  );
};

// Add CSS animation for connection indicator
const style = document.createElement('style');
style.textContent = `
  @keyframes pulse {
    0% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
    100% {
      opacity: 1;
    }
  }
`;
document.head.appendChild(style);

export default Dashboard;