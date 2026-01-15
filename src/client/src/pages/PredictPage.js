import React, { Component } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import LayoutPage from './LayoutPage';
import { Table, Tooltip, notification, Upload, Spin, Button, Form, Select, Menu, Modal, Divider, Card, Row, Col, Statistic, Tag, Space, Typography } from 'antd';
import { UploadOutlined, CheckCircleOutlined, WarningOutlined, ClockCircleOutlined, PlayCircleOutlined, StopOutlined, LockOutlined, FileTextOutlined, SendOutlined, UserOutlined, DatabaseOutlined } from "@ant-design/icons";
import { connect } from "react-redux";
import { useUserRole } from '../hooks/useUserRole';
import { Pie, Bar } from '@ant-design/plots';
import {
  FORM_LAYOUT,
  SERVER_URL,
  MAX_PCAP_SIZE_BYTES,
  MAX_PCAP_SIZE_MB,
} from "../constants";
import { getUserHeaders, fetchWithAuth } from '../utils/fetchWithAuth';
import {
  requestApp,
  requestBuildConfigModel,
  requestAllReports,
  requestAllModels,
  requestPredict,
  requestPredictStatus,
  requestRunLime,
} from "../actions";
import {
  requestPredictStats,
  requestPredictionAttack,
  requestAssistantExplainFlow,
  requestPredictStatus as apiRequestPredictStatus,
  requestPredictOnlineStart,
  requestPredictOnlineStatus,
  requestPredictOnlineStop,
} from "../api";
import {
  getFilteredModelsOptions,
  getLastPath,
} from "../utils";
import { handleMitigationAction, handleBulkMitigationAction } from '../utils/mitigation';
import { buildAttackTable } from '../utils/attacksTable';

let isModelIdPresent = getLastPath() !== "offline" && getLastPath() !== "online" && getLastPath() !== "predict";

class PredictPage extends Component {
  constructor(props) {
    super(props);
    this.state = {
      // Mode selection
      mode: 'offline', // 'offline' or 'online'
      
      // Offline mode
      modelId: null,
      testingPcapFile: null,
      testingDataset: null,
      pcapFiles: [],
      wasUploaded: false,
      
      // Online mode
      interface: null,
      interfacesOptions: [],
      isRunningOnline: false,
      
      // Shared state
      isRunning: (props.predictStatus && props.predictStatus.isRunning) ? props.predictStatus.isRunning : false,
      isMMTRunning: (props.mmtStatus && props.mmtStatus.isRunning) ? props.mmtStatus.isRunning : false,
      currentJobId: null, // Job ID for queued predictions
      currentPredictionId: null, // Prediction ID for results
      predictStats: null,
      attackCsv: null,
      attackRows: [],
      attackColumns: [],
      attackFlowColumns: [],
      mitigationColumns: [],
      attackPagination: { current: 1, pageSize: 10 },
      
      // Online aggregation
      aggregateNormal: 0,
      aggregateMalicious: 0,
      hasResultsShown: false,
      lastShownPredictionId: null,
      lastStatsSignature: null,
      loadedPredictionIds: [],
      
      // Modals
      limeModalVisible: false,
      limeValues: [],
      assistantModalVisible: false,
      assistantText: '',
      assistantLoading: false,
      assistantTokenInfo: null,
    };
    this.handleButtonStart = this.handleButtonStart.bind(this);
    this.handleButtonStop = this.handleButtonStop.bind(this);
    this.pollOnlineResults = this.pollOnlineResults.bind(this);
  }

  // Extract flow details (IPs, ports, rates, sessionId) from a record
  computeFlowDetails = (record) => {
    const keyList = Object.keys(record).filter(k => k !== 'key');
    
    // First try to extract IPs from ip.pkts_per_flow column (format: "['ip1', 'ip2']")
    let srcIp = null;
    let dstIp = null;
    const pktsPerFlow = record['ip.pkts_per_flow'];
    if (pktsPerFlow && typeof pktsPerFlow === 'string') {
      const matches = pktsPerFlow.match(/\['([^']+)',\s*'([^']+)'\]/);
      if (matches && matches.length === 3) {
        srcIp = matches[1];
        dstIp = matches[2];
      }
    }
    
    // Fallback: Try to find separate IP columns if not found in ip.pkts_per_flow
    if (!srcIp || !dstIp) {
      const findKey = (patterns) => keyList.find(k => patterns.some(p => p.test(k)));
      const srcKey = findKey([
        /src.*ip/i, /source.*ip/i, /^ip[_-]?src$/i, /^src[_-]?ip$/i, /(src|source).*addr/i, /^saddr$/i
      ]);
      const dstKey = findKey([
        /dst.*ip/i, /dest.*ip/i, /destination.*ip/i, /^ip[_-]?dst$/i, /^dst[_-]?ip$/i, /(dst|dest|destination).*addr/i, /^daddr$/i
      ]);
      const combinedIpKey = (!srcKey && !dstKey) ? findKey([/^ip$/i, /ip.*pair/i, /ip.*addr/i, /address/i]) : null;
      
      if (srcKey && !srcIp) srcIp = record[srcKey];
      if (dstKey && !dstIp) dstIp = record[dstKey];
      
      // Try extracting from combined IP key
      if (combinedIpKey && (!srcIp || !dstIp)) {
        const text = String(record[combinedIpKey] || '');
        const ipv4s = text.match(/(?:\d{1,3}\.){3}\d{1,3}/g) || [];
        if (!srcIp) srcIp = ipv4s[0] || null;
        if (!dstIp) dstIp = ipv4s[1] || null;
      }
    }
    
    const sessionId = record['ip.session_id'] || record['session_id'] || null;
    const dport = record['dport_g'] ?? record['dport_le'] ?? record['dport'] ?? null;
    const pktsRate = record['pkts_rate'] ?? null;
    const byteRate = record['byte_rate'] ?? null;
    
    console.log('[computeFlowDetails]', { srcIp, dstIp, dport, pktsPerFlow: pktsPerFlow?.substring(0, 50) });
    
    return { srcIp, dstIp, sessionId, dport, pktsRate, byteRate };
  }

  // Strict IPv4 validation (each octet 0-255)
  isValidIPv4(ip) {
    if (typeof ip !== 'string') return false;
    const m = ip.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
    if (!m) return false;
    return m.slice(1).every(o => {
      const n = Number(o);
      return n >= 0 && n <= 255 && String(n) === String(Number(o));
    });
  }

  // Validate port number (1-65535, integer)
  isValidPort(port) {
    if (port === null || port === undefined) return false;
    const portNum = Number(port);
    if (isNaN(portNum)) return false;
    // Must be integer between 1 and 65535
    return portNum >= 1 && portNum <= 65535 && portNum === Math.floor(portNum);
  }

  componentDidMount() {
    const path = getLastPath();
    
    // Determine initial mode from URL (with permission check)
    if (path === 'online') {
      // Only allow online mode if user has permission
      if (this.props.canPerformOnlineActions) {
        this.setState({ mode: 'online' });
      } else {
        // Redirect non-admin users to offline mode
        this.setState({ mode: 'offline' });
        window.history.replaceState(null, '', '/predict/offline');
        notification.warning({ message: 'Access Denied', description: 'Administrator privileges required for online predictions. Switched to offline mode.', placement: 'topRight' });
      }
    } else if (path === 'offline') {
      this.setState({ mode: 'offline' });
    } else if (path === 'predict' || !path) {
      // Default to offline mode if no specific mode in URL
      this.setState({ mode: 'offline' });
    }
    
    // Only set modelId if path is not a reserved word (online/offline/predict)
    if (isModelIdPresent && path !== 'online' && path !== 'offline' && path !== 'predict') {
      this.setState({ modelId: path });
    }
    this.props.fetchApp();
    this.props.fetchAllReports();
    this.props.fetchAllModels();
    this.fetchInterfacesAndSetOptions();
    // Load PCAP files only after auth is loaded so x-user-id is sent
    if (this.props.isAuthLoaded) {
      this.fetchPcapFiles();
    }
    
    // Load previously uploaded PCAPs for reuse
    try {
      const raw = localStorage.getItem('uploadedPcaps');
      const list = raw ? JSON.parse(raw) : [];
      if (Array.isArray(list)) {
        this.setState({ uploadedPcaps: list });
      }
    } catch (e) {
      // ignore storage errors
    }
    // If navigated from Feature Extraction with a pending PCAP, pre-select it
    // Server now handles MMT analysis automatically when prediction is started
    try {
      const pending = localStorage.getItem('pendingPredictOfflinePcap');
      if (pending) {
        this.setState({ mode: 'offline', testingPcapFile: pending, predictStats: null });
        localStorage.removeItem('pendingPredictOfflinePcap');
        localStorage.removeItem('pendingPredictOfflineReportId');
      }
    } catch (e) {
      // ignore storage errors
    }
  }
  
  componentWillUnmount() {
    if (this.statusTimer) clearInterval(this.statusTimer);
    if (this.predictTimer) clearInterval(this.predictTimer);
    if (this.intervalId) clearInterval(this.intervalId);
    if (this.chartRefreshTimer) clearInterval(this.chartRefreshTimer);
  }
  
  async fetchInterfacesAndSetOptions() {
    let interfacesOptions = [];
    try {
      const url = `${SERVER_URL}/api/predict/interfaces`;
      const response = await fetch(url);
      const data = await response.json();
      if (data.error) throw data.error;
      const interfaces = data.interfaces;
      interfacesOptions = interfaces.map(i => {
        const dev = String(i).split(/\s*-\s*/)[0];
        return { label: i, value: dev };
      });
    } catch (error) {
      console.error('Error:', error);
    }
    this.setState({ interfacesOptions });
  }

  async fetchPcapFiles() {
    try {
      const res = await fetchWithAuth(`${SERVER_URL}/api/pcaps`, {}, this.props.userRole);
      if (!res.ok) return;
      const data = await res.json();
      this.setState({ pcapFiles: data.pcaps || [] });
    } catch (e) {
      // ignore
    }
  }

  // Retrieve cached report id mapped to a given PCAP filename from localStorage
  getCachedReportForPcap = (pcapName) => {
    try {
      const raw = localStorage.getItem('pcapToReport');
      const map = raw ? JSON.parse(raw) : {};
      if (map && typeof map === 'object' && !Array.isArray(map)) {
        return map[pcapName] || null;
      }
    } catch (e) {
      // ignore
    }
    return null;
  }

  beforeUploadPcap = (file) => {
    const hasValidExtension = file && (file.name.endsWith('.pcap') || file.name.endsWith('.pcapng') || file.name.endsWith('.cap'));
    if (!hasValidExtension) {
      notification.error({
        message: 'Invalid File Type',
        description: `${file?.name || 'File'} is not a valid PCAP file. Only .pcap, .pcapng, and .cap files are allowed.`,
        placement: 'topRight',
        duration: 2,
      });
      return Upload.LIST_IGNORE;
    }
    if (file.size > MAX_PCAP_SIZE_BYTES) {
      notification.error({
        message: 'File Too Large',
        description: `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds the maximum allowed size of ${MAX_PCAP_SIZE_MB} MB. Please use a smaller file.`,
        placement: 'topRight',
        duration: 2,
      });
      return Upload.LIST_IGNORE;
    }
    return true;
  }

  handleUploadPcap = async (info, typePcap) => {
    const { status, name } = info.file;
    console.log({ status, name });

    if (status === 'uploading') {
      console.log(`Uploading ${name}`);
    } else if (status === 'done') {
      console.log(`Uploaded successfully ${name}`);
    } else if (status === 'error') {
      notification.error({ message: 'Upload Failed', description: `${name} upload failed.`, placement: 'topRight' });
    }
  };

  processUploadPcap = async ({ file, onProgress, onSuccess, onError }) => {
    const formData = new FormData();
    formData.append('pcapFile', file);

    try {
      const response = await fetch(`${SERVER_URL}/api/pcaps`, {
        method: 'POST',
        body: formData,
        headers: getUserHeaders(this.props.userRole),
      });

      if (response.ok) {
        const data = await response.json();
        onSuccess(data, response);
        this.setState({ testingPcapFile: data.pcapFile, wasUploaded: true });
        // Refresh PCAP files list
        this.fetchPcapFiles();
        const isDuplicate = data.alreadyExisted || false;
        notification.success({
          message: isDuplicate ? 'PCAP Already Exists' : 'PCAP Uploaded',
          description: isDuplicate 
            ? `File "${data.pcapFile}" already exists. Using existing file for prediction.`
            : `File "${data.pcapFile}" uploaded successfully and ready for prediction.`,
          placement: 'topRight',
        });
      } else {
        let errorText = await response.text();
        try { const j = JSON.parse(errorText); errorText = j.error || j.message || errorText; } catch {}
        onError(new Error(errorText));
        notification.error({
          message: 'Upload Failed',
          description: errorText,
          placement: 'topRight',
          duration: 2,
        });
      }
    } catch (error) {
      onError(error);
      notification.error({
        message: 'Upload Failed',
        description: error.message || String(error),
        placement: 'topRight',
        duration: 2,
      });
    }
  }

  handlePredictOffline = async () => {
    const {
      modelId,
      testingPcapFile,
      isRunning,
    } = this.state;

    let fetchModelId = isModelIdPresent ? getLastPath() : modelId;

    if (!isRunning) {
      // Set isRunning immediately to disable button and show loading spinner
      this.setState({ isRunning: true });

      if (!testingPcapFile) {
        this.setState({ isRunning: false });
        notification.error({ message: 'Missing Input', description: 'Please upload a PCAP file first', placement: 'topRight' });
        return;
      }

      // Use simplified API - server handles MMT analysis automatically
      const { requestPredictOfflineSimplified, requestPredictJobStatus } = require('../api');

      try {
        console.log('Starting simplified prediction for:', testingPcapFile, 'with model:', fetchModelId);

        const queueResponse = await requestPredictOfflineSimplified(fetchModelId, testingPcapFile, true);

        console.log('Prediction queued:', queueResponse);
        this.setState({
          currentJobId: queueResponse.jobId,
          currentPredictionId: queueResponse.predictionId
        });

        notification.success({
          message: 'Prediction Job Queued',
          description: 'Server is analyzing PCAP and running prediction...',
          placement: 'topRight',
          duration: 3,
        });

        // Poll job status
        this.intervalId = setInterval(async () => {
          try {
            const jobStatus = await requestPredictJobStatus(this.state.currentJobId);
            console.log('Job status:', jobStatus.status, 'Progress:', jobStatus.progress);

            if (jobStatus.status === 'completed') {
              clearInterval(this.intervalId);
              this.intervalId = null;
              this.setState({ isRunning: false });

              // Load prediction results
              const predictionId = this.state.currentPredictionId;
              console.log('Prediction completed, loading results for:', predictionId);

              try {
                const { requestPredictStats, requestPredictionAttack } = require('../api');
                const predictStats = await requestPredictStats(predictionId);
                console.log('Fetched predictStats:', predictStats);

                let attackCsv = null;
                try {
                  attackCsv = await requestPredictionAttack(predictionId);
                } catch (e) {
                  console.warn('No attack CSV available:', e.message);
                }

                let attackRows = [];
                let attackFlowColumns = [];
                let mitigationColumns = [];

                if (attackCsv) {
                  const built = buildAttackTable({
                    csvString: attackCsv,
                    onAction: (key, record) => this.onMitigationAction(key, record),
                    buildMenu: (record, onAction) => {
                      const { srcIp, dstIp, dport } = this.computeFlowDetails(record);
                      const validSrc = this.isValidIPv4(srcIp);
                      const validDst = this.isValidIPv4(dstIp);
                      const validPort = this.isValidPort(dport);
                      const { userRole } = this.props;
                      const assistantDisabled = !userRole?.isSignedIn || userRole?.tokenLimitReached;
                      return (
                        <Menu onClick={({ key }) => onAction && onAction(key, record)}>
                          <Menu.Item key="explain-gpt" disabled={assistantDisabled}>
                            Ask Assistant
                            {assistantDisabled && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                          </Menu.Item>
                          <Menu.Item key="explain-shap">Explain (XAI SHAP)</Menu.Item>
                          <Menu.Item key="explain-lime">Explain (XAI LIME)</Menu.Item>
                          <Menu.Divider />
                          <Menu.Item key="block-src-ip" disabled={!validSrc}>
                            {`Block source IP${validSrc ? ` ${srcIp}` : ''}`}
                          </Menu.Item>
                          <Menu.Item key="block-dst-ip" disabled={!validDst}>
                            {`Block destination IP${validDst ? ` ${dstIp}` : ''}`}
                          </Menu.Item>
                          {validPort && (
                            <>
                              <Menu.Divider />
                              <Menu.Item key="block-dst-port">
                                {`Block destination port ${dport}/tcp`}
                              </Menu.Item>
                            </>
                          )}
                          <Menu.Divider />
                          <Menu.Item key="drop-session" disabled={!(validSrc || validDst)}>
                            {`Drop session${validSrc ? ` ${srcIp}` : validDst ? ` ${dstIp}` : ''}`}
                          </Menu.Item>
                          <Menu.Divider />
                          <Menu.Item key="send-nats" disabled={!userRole?.isAdmin}>
                            Send flow to NATS
                            {!userRole?.isAdmin && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                          </Menu.Item>
                        </Menu>
                      );
                    }
                  });
                  attackRows = built.rows;
                  attackFlowColumns = built.flowColumns;
                  mitigationColumns = built.mitigationColumns;
                }

                this.setState({
                  predictStats,
                  attackCsv,
                  attackRows,
                  attackFlowColumns,
                  mitigationColumns
                });

                notification.success({
                  message: 'Success',
                  description: 'Prediction completed successfully!',
                  placement: 'topRight',
                });
              } catch (error) {
                console.error('Error loading prediction results:', error);
                notification.error({ message: 'Error', description: 'Failed to load prediction results: ' + error.message, placement: 'topRight' });
              }
            } else if (jobStatus.status === 'failed') {
              clearInterval(this.intervalId);
              this.intervalId = null;
              this.setState({ isRunning: false });
              console.error('Prediction failed:', jobStatus.failedReason);
              notification.error({ message: 'Prediction Failed', description: jobStatus.failedReason, placement: 'topRight' });
            }
          } catch (error) {
            console.error('Error polling job status:', error);
          }
        }, 2000);
      } catch (error) {
        console.error('Error starting prediction:', error);
        this.setState({ isRunning: false });
        notification.error({ message: 'Prediction Error', description: error.message || 'Failed to start prediction', placement: 'topRight' });
      }
    }
  }

  handleTablePredictStats = (csvData) => {
    // Handle empty or malformed CSV data
    if (!csvData || typeof csvData !== 'string') {
      return { tableConfig: { dataSource: [], columns: [], pagination: false }, normalFlows: 0, maliciousFlows: 0 };
    }

    const lines = csvData.trim().split('\n').filter(line => line.trim().length > 0);
    if (lines.length === 0) {
      return { tableConfig: { dataSource: [], columns: [], pagination: false }, normalFlows: 0, maliciousFlows: 0 };
    }

    // stats.csv has no header - just data rows (normal,attack,total)
    // Use the last line for cumulative stats
    const lastLine = lines[lines.length - 1];
    const values = lastLine.split(',');
    const normalFlows = parseInt(values[0], 10) || 0;
    const maliciousFlows = parseInt(values[1], 10) || 0;
    const dataSource = [
      {
        key: 'data',
        "Normal flows": values[0] || '0',
        "Malicious flows": values[1] || '0',
        "Total flows": values[2] || '0'
      }
    ];
    const columns = [
      {
        title: 'Normal flows',
        dataIndex: 'Normal flows',
        align: 'center',
      },
      {
        title: 'Malicious flows',
        dataIndex: 'Malicious flows',
        align: 'center',
      },
      {
        title: 'Total flows',
        dataIndex: 'Total flows',
        align: 'center',
      }
    ];
    const tableConfig = {
      dataSource: dataSource,
      columns: columns,
      pagination: false
    };

    return { tableConfig, normalFlows, maliciousFlows };
  }

  async componentDidUpdate(prevProps, prevState) {
    // When auth becomes loaded, refresh PCAP list with user headers
    if (!prevProps.isAuthLoaded && this.props.isAuthLoaded) {
      this.fetchPcapFiles();
    }
    if (this.props.app !== prevProps.app && !isModelIdPresent) {
      this.setState({ modelId: null });
    }

    const prevPS = prevProps && prevProps.predictStatus ? prevProps.predictStatus : {};
    const currPS = this.props && this.props.predictStatus ? this.props.predictStatus : {};
    if ((prevPS.isRunning || false) !== (currPS.isRunning || false)) {
      console.log('isRunning has been changed');
      this.setState({ isRunning: !!currPS.isRunning });
      if (!currPS.isRunning) {
        // Clear all prediction polling timers
        if (this.intervalId) {
          clearInterval(this.intervalId);
          this.intervalId = null;
        }
        if (this.predictTimer) {
          clearInterval(this.predictTimer);
          this.predictTimer = null;
        }
        
        const { mode } = this.state;
        if (mode === 'offline') {
          notification.success({
            message: 'Success',
            description: 'Make predictions successfully!',
            placement: 'topRight',
          });
          // Don't clear dataset/pcap - keep them to show results
          // User can manually clear them if they want to run another prediction
        } else {
          notification.success({ message: 'Completed', description: 'Online window prediction completed', placement: 'topRight' });
        }
        
        const lastPredictId = currPS.lastPredictedId || '';
        if (lastPredictId) {
          if (mode === 'online') {
            await this.appendAttackRowsFromPredictionId(lastPredictId);
          } else {
            // Offline mode: Fetch stats and attacks
            const predictStats = await requestPredictStats(lastPredictId);
            console.log('[componentDidUpdate] Fetched predictStats:', predictStats);
            let attackCsv = null;
            try {
              attackCsv = await requestPredictionAttack(lastPredictId);
            } catch (e) {
              console.warn('No attack CSV available:', e.message);
            }
            let attackRows = [];
            let attackFlowColumns = [];
            let mitigationColumns = [];
            if (attackCsv) {
              const built = buildAttackTable({
                csvString: attackCsv,
                onAction: (key, record) => this.onMitigationAction(key, record),
                buildMenu: (record, onAction) => {
                  const { srcIp, dstIp, dport } = this.computeFlowDetails(record);
                  const validSrc = this.isValidIPv4(srcIp);
                  const validDst = this.isValidIPv4(dstIp);
                  const validPort = this.isValidPort(dport);
                  const { userRole } = this.props;
                  const assistantDisabled = !userRole?.isSignedIn || userRole?.tokenLimitReached;
                  const natsDisabled = !userRole?.isAdmin;
                  return (
                    <Menu onClick={({ key }) => onAction && onAction(key, record)}>
                      <Menu.Item key="explain-gpt" disabled={assistantDisabled}>
                        Ask Assistant
                        {assistantDisabled && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                      </Menu.Item>
                      <Menu.Item key="explain-shap">Explain (XAI SHAP)</Menu.Item>
                      <Menu.Item key="explain-lime">Explain (XAI LIME)</Menu.Item>
                      <Menu.Divider />
                      <Menu.Item key="block-src-ip" disabled={!validSrc}>
                        {`Block source IP${validSrc ? ` ${srcIp}` : ''}`}
                      </Menu.Item>
                      <Menu.Item key="block-dst-ip" disabled={!validDst}>
                        {`Block destination IP${validDst ? ` ${dstIp}` : ''}`}
                      </Menu.Item>
                      {validPort && (
                        <>
                          <Menu.Divider />
                          <Menu.Item key="block-dst-port">
                            {`Block destination port ${dport}/tcp`}
                          </Menu.Item>
                          {validSrc && (
                            <Menu.Item key="block-ip-port-src">
                              {`Block ${srcIp}:${dport}/tcp`}
                            </Menu.Item>
                          )}
                          {validDst && (
                            <Menu.Item key="block-ip-port-dst">
                              {`Block ${dstIp}:${dport}/tcp`}
                            </Menu.Item>
                          )}
                        </>
                      )}
                      <Menu.Divider />
                      <Menu.Item key="drop-session" disabled={!(validSrc || validDst)}>
                        {`Drop session${validSrc ? ` ${srcIp}` : validDst ? ` ${dstIp}` : ''}`}
                      </Menu.Item>
                      {validPort && validSrc && (
                        <Menu.Item key="rate-limit-src">
                          {`Rate-limit source ${srcIp}:${dport}/tcp`}
                        </Menu.Item>
                      )}
                      <Menu.Divider />
                      <Menu.Item key="send-nats" disabled={natsDisabled}>
                        Send flow to NATS
                        {natsDisabled && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                      </Menu.Item>
                    </Menu>
                  );
                }
              });
              attackRows = built.rows;
              attackFlowColumns = built.flowColumns;
              mitigationColumns = built.mitigationColumns;
            }
            console.log('[componentDidUpdate] Setting state with predictStats:', predictStats);
            this.setState({ predictStats, attackCsv, attackRows, attackFlowColumns, mitigationColumns });
          }
        }
      }
    }
    
    // Online mode: append rows whenever a new prediction id is produced
    if (this.state.mode === 'online') {
      const prevLastId = prevProps.predictStatus && prevProps.predictStatus.lastPredictedId;
      const currLastId = this.props.predictStatus && this.props.predictStatus.lastPredictedId;
      if (currLastId && currLastId !== prevLastId) {
        await this.appendAttackRowsFromPredictionId(currLastId);
      }
    }
  }
  
  // Online mode: Append malicious rows for a finished prediction id
  appendAttackRowsFromPredictionId = async (predictionId) => {
    if (!predictionId) return;
    // Avoid duplicate loads
    if ((this.state.loadedPredictionIds || []).includes(predictionId)) return;
    try {
      const predictStats = await requestPredictStats(predictionId);
      // Handle empty or malformed predictStats
      // stats.csv has no header - just data rows (normal,attack,total)
      const lines = (predictStats || '').trim().split('\n').filter(line => line.trim().length > 0);
      const lastLine = lines.length > 0 ? lines[lines.length - 1] : '0,0,0';
      const values = lastLine.split(',');
      const normal = parseInt(values[0], 10) || 0;
      const malicious = parseInt(values[1], 10) || 0;
      const nextSignature = String(predictStats || '').trim();
      let attackCsv = null;
      try {
        attackCsv = await requestPredictionAttack(predictionId);
      } catch (e) {
        // ignore
      }
      let built = null;
      if (attackCsv) {
        built = buildAttackTable({
          csvString: attackCsv,
          onAction: (key, record) => this.onMitigationAction(key, record),
          buildMenu: (record, onAction) => {
            const { srcIp, dstIp, dport } = this.computeFlowDetails(record);
            const validSrc = this.isValidIPv4(srcIp);
            const validDst = this.isValidIPv4(dstIp);
            const validPort = this.isValidPort(dport);
            const { userRole } = this.props;
            const assistantDisabled = !userRole?.isSignedIn || userRole?.tokenLimitReached;
            return (
              <Menu onClick={({ key }) => onAction && onAction(key, record)}>
                <Menu.Item key="explain-gpt" disabled={assistantDisabled}>
                  Ask Assistant
                  {assistantDisabled && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                </Menu.Item>
                <Menu.Item key="explain-shap" disabled>Explain (XAI SHAP)</Menu.Item>
                <Menu.Item key="explain-lime" disabled>Explain (XAI LIME)</Menu.Item>
                <Menu.Divider />
                <Menu.Item key="block-src-ip" disabled={!validSrc}>{`Block source IP${validSrc ? ` ${srcIp}` : ''}`}</Menu.Item>
                <Menu.Item key="block-dst-ip" disabled={!validDst}>{`Block destination IP${validDst ? ` ${dstIp}` : ''}`}</Menu.Item>
                {validPort && (
                  <>
                    <Menu.Divider />
                    <Menu.Item key="block-dst-port">{`Block destination port ${dport}/tcp`}</Menu.Item>
                    {validSrc && <Menu.Item key="block-ip-port-src">{`Block ${srcIp}:${dport}/tcp`}</Menu.Item>}
                    {validDst && <Menu.Item key="block-ip-port-dst">{`Block ${dstIp}:${dport}/tcp`}</Menu.Item>}
                  </>
                )}
                <Menu.Divider />
                <Menu.Item key="drop-session" disabled={!(validSrc || validDst)}>{`Drop session${validSrc ? ` ${srcIp}` : validDst ? ` ${dstIp}` : ''}`}</Menu.Item>
                {validPort && validSrc && <Menu.Item key="rate-limit-src">{`Rate-limit source ${srcIp}:${dport}/tcp`}</Menu.Item>}
                <Menu.Divider />
                <Menu.Item key="send-nats" disabled={!userRole?.isAdmin}>
                  Send flow to NATS
                  {!userRole?.isAdmin && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                </Menu.Item>
              </Menu>
            );
          }
        });
        if (built && Array.isArray(built.rows)) {
          built.rows = built.rows.map((r, idx) => ({
            ...r,
            __predictionId: predictionId,
            __rowUid: `${predictionId}-${r.key || (idx + 1)}`,
          }));
        }
        if (built && Array.isArray(built.flowColumns)) {
          built.flowColumns = built.flowColumns.filter(col => {
            const di = String(col.dataIndex || '').toLowerCase();
            const isSessionCol = (di === 'ip.session_id' || di === 'session_id' || di.endsWith('session_id'));
            const isInternal = di.startsWith('__');
            const isMalwareCol = di.includes('malware') || di.includes('malicious');
            return !(isSessionCol || isInternal || isMalwareCol);
          });
        }
      }
      this.setState(prev => {
        const newRows = built?.rows || [];
        const nextAttackRows = newRows.length > 0 ? [...prev.attackRows, ...newRows] : prev.attackRows;
        const attackFlowColumns = (prev.attackFlowColumns && prev.attackFlowColumns.length > 0) ? prev.attackFlowColumns : (built?.flowColumns || []);
        const mitigationColumns = (prev.mitigationColumns && prev.mitigationColumns.length > 0) ? prev.mitigationColumns : (built?.mitigationColumns || []);
        const canUpdateCharts = (predictionId !== prev.lastShownPredictionId) && (nextSignature !== prev.lastStatsSignature);
        return {
          predictStats: canUpdateCharts ? predictStats : prev.predictStats,
          aggregateNormal: canUpdateCharts ? (prev.aggregateNormal + normal) : prev.aggregateNormal,
          aggregateMalicious: canUpdateCharts ? (prev.aggregateMalicious + malicious) : prev.aggregateMalicious,
          attackCsv,
          attackRows: nextAttackRows,
          attackFlowColumns,
          mitigationColumns,
          hasResultsShown: prev.hasResultsShown || (!!predictStats),
          lastShownPredictionId: canUpdateCharts ? predictionId : prev.lastShownPredictionId,
          lastStatsSignature: canUpdateCharts ? nextSignature : prev.lastStatsSignature,
          loadedPredictionIds: [...(prev.loadedPredictionIds || []), predictionId],
        };
      });
    } catch (e) {
      console.error('Failed to append rows for prediction:', predictionId, e);
    }
  }

  onMitigationAction = (key, record) => {
    // Derive common fields similar to PredictOnlinePage
    const { srcIp, dstIp, sessionId, dport, pktsRate, byteRate } = this.computeFlowDetails(record);

    if (key === 'explain-gpt') {
      const modelId = this.state.modelId;
      const predictionId = this.props.predictStatus?.lastPredictedId || '';
      const { userRole } = this.props;
      if (!modelId) return;
      this.setState({ assistantModalVisible: true, assistantLoading: true, assistantText: '', assistantTokenInfo: null });
      requestAssistantExplainFlow({
        flowRecord: record,
        modelId,
        predictionId,
        extra: { srcIp, dstIp, sessionId, dport, pktsRate, byteRate },
        userId: userRole?.userId,
        isAdmin: userRole?.isAdmin,
      }).then((resp) => {
        this.setState({ assistantText: resp.text || '', assistantLoading: false, assistantTokenInfo: resp.tokenUsage });
        
        // Show token usage notification
        if (resp.tokenUsage) {
          const { thisRequest, remaining, limit, percentUsed } = resp.tokenUsage;
          if (limit === Infinity) {
            notification.success({
              message: 'AI Explanation Generated',
              description: `Tokens used: ${thisRequest} - Unlimited (Admin)`,
              placement: 'topRight',
              duration: 2,
            });
          } else {
            const color = percentUsed >= 90 ? 'warning' : 'success';
            const remainingStr = remaining != null && remaining !== Infinity ? remaining.toLocaleString() : '0';
            const limitStr = limit != null && limit !== Infinity ? limit.toLocaleString() : '0';
            notification[color]({
              message: 'AI Explanation Generated',
              description: `Tokens used: ${thisRequest} - Remaining: ${remainingStr}/${limitStr} (${percentUsed}% used)`,
              placement: 'topRight',
              duration: 5,
            });
          }
        }
      }).catch((e) => {
        this.setState({ assistantText: `Error: ${e.message || String(e)}`, assistantLoading: false });
        notification.error({
          message: 'AI Assistant Error',
          description: e.message || String(e),
          placement: 'topRight',
        });
      });
      return;
    }

    if (key === 'explain-lime' || key === 'explain-shap') {
      const modelId = this.state.modelId;
      const predictionId = this.props.predictStatus?.lastPredictedId || '';
      if (modelId && sessionId) {
        const qp = new URLSearchParams({ sampleId: String(sessionId) });
        // Always set predictionId or a flag to indicate this is from prediction context (not test dataset)
        if (predictionId) {
          qp.set('predictionId', predictionId);
        } else {
          // Use a flag to indicate this is from prediction flow (no ground truth available)
          qp.set('fromPrediction', 'true');
        }
        const base = key === 'explain-lime' ? '/xai/lime/' : '/xai/shap/';
        const target = `${base}${encodeURIComponent(modelId)}?${qp.toString()}`;
        window.location.href = target;
      }
      return;
    }

    handleMitigationAction({
      actionKey: key,
      srcIp,
      dstIp,
      sessionId,
      dport,
      pktsRate,
      byteRate,
      isValidIPv4: this.isValidIPv4,
      flowRecord: record
    });
  }

  // Online mode: Poll status and process slices
  /**
   * Poll online prediction results (replaces old tcpdump workflow)
   */
  async pollOnlineResults() {
    // Check if we should still be polling
    if (!this.state.isRunningOnline) {
      return;
    }

    try {
      // Get online prediction status
      const status = await requestPredictOnlineStatus();

      if (!status.isRunning) {
        // Prediction stopped
        this.setState({ isRunningOnline: false });
        if (this.statusTimer) {
          clearInterval(this.statusTimer);
          this.statusTimer = null;
        }
        if (this.chartRefreshTimer) {
          clearInterval(this.chartRefreshTimer);
          this.chartRefreshTimer = null;
        }
        return;
      }

      const { currentPredictionId } = this.state;
      if (!currentPredictionId) {
        return;
      }

      // Fetch prediction results using authenticated API
      const predictStats = await requestPredictStats(currentPredictionId);

      if (predictStats) {
        const rows = predictStats.trim().split('\n').filter(r => r.trim().length > 0);
        // stats.csv format: no header, just data rows (normal_count,attack_count,total_count)
        if (rows.length > 0) {
          const lastRow = rows[rows.length - 1]; // Get the latest stats
          const cols = lastRow.split(',');
          if (cols.length >= 2) {
            const normal = parseInt(cols[0]) || 0;
            const malicious = parseInt(cols[1]) || 0;

            this.setState({
              aggregateNormal: normal,
              aggregateMalicious: malicious,
              predictStats,
              hasResultsShown: true
            });
          }
        }
      }

      // Fetch attack details using authenticated API
      const attackCsv = await requestPredictionAttack(currentPredictionId);
      if (attackCsv && attackCsv.trim().length > 0) {
        // Build attack table
        const { rows: attackRows, flowColumns: attackFlowColumns, mitigationColumns } = buildAttackTable({
          csvString: attackCsv,
          onAction: (key, record) => this.onMitigationAction(key, record),
          buildMenu: (record, onAction) => {
            const { srcIp, dstIp, dport } = this.computeFlowDetails(record);
            const validSrc = this.isValidIPv4(srcIp);
            const validDst = this.isValidIPv4(dstIp);
            const validPort = this.isValidPort(dport);
            const { userRole } = this.props;
            const assistantDisabled = !userRole?.isSignedIn || userRole?.tokenLimitReached;
            return (
              <Menu onClick={({ key }) => onAction && onAction(key, record)}>
                <Menu.Item key="explain-gpt" disabled={assistantDisabled}>
                  Ask Assistant
                  {assistantDisabled && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                </Menu.Item>
                <Menu.Item key="explain-shap" disabled>Explain (XAI SHAP)</Menu.Item>
                <Menu.Item key="explain-lime" disabled>Explain (XAI LIME)</Menu.Item>
                <Menu.Divider />
                <Menu.Item key="block-src-ip" disabled={!validSrc}>{`Block source IP${validSrc ? ` ${srcIp}` : ''}`}</Menu.Item>
                <Menu.Item key="block-dst-ip" disabled={!validDst}>{`Block destination IP${validDst ? ` ${dstIp}` : ''}`}</Menu.Item>
                {validPort && (
                  <>
                    <Menu.Divider />
                    <Menu.Item key="block-dst-port">{`Block destination port ${dport}/tcp`}</Menu.Item>
                  </>
                )}
                <Menu.Divider />
                <Menu.Item key="drop-session" disabled={!(validSrc || validDst)}>{`Drop session${validSrc ? ` ${srcIp}` : validDst ? ` ${dstIp}` : ''}`}</Menu.Item>
                <Menu.Divider />
                <Menu.Item key="send-nats" disabled={!userRole?.isAdmin}>
                  Send flow to NATS
                  {!userRole?.isAdmin && <LockOutlined style={{ marginLeft: 8, fontSize: '11px', color: '#ff4d4f' }} />}
                </Menu.Item>
              </Menu>
            );
          }
        });
        
        this.setState({ 
          attackCsv, 
          attackRows, 
          attackFlowColumns, 
          mitigationColumns,
          hasResultsShown: true 
        });
      }

    } catch (e) {
      console.warn('Failed to poll online results:', e.message);
    }
  }

  /**
   * Start online prediction using integrated MMT-probe approach
   */
  async handleButtonStart() {
    const { modelId, interface: iface } = this.state;

    // Security check
    if (!this.props.canPerformOnlineActions) {
      notification.error({ message: 'Access Denied', description: 'Administrator privileges required for online predictions', placement: 'topRight' });
      this.setState({ mode: 'offline' });
      return;
    }

    if (!modelId) {
      notification.warning({ message: 'Missing Selection', description: 'Please select a model first', placement: 'topRight' });
      return;
    }

    if (!iface) {
      notification.warning({ message: 'Missing Selection', description: 'Please select a network interface', placement: 'topRight' });
      return;
    }

    try {
      // Start integrated online prediction
      const result = await requestPredictOnlineStart(modelId, iface);

      if (result.success) {
        notification.success({ message: 'Started', description: `Online prediction started on ${iface}`, placement: 'topRight' });
        this.setState({
          isRunningOnline: true,
          currentPredictionId: result.predictionId,
          aggregateNormal: 0,
          aggregateMalicious: 0,
          attackCsv: null,
          attackRows: [],
          attackFlowColumns: [],
          mitigationColumns: [],
        });

        // Start polling for results (every 10 seconds to reduce server load)
        if (this.statusTimer) clearInterval(this.statusTimer);
        this.statusTimer = setInterval(this.pollOnlineResults, 10000);

        // Chart refresh timer
        if (this.chartRefreshTimer) clearInterval(this.chartRefreshTimer);
        this.chartRefreshTimer = setInterval(() => this.forceUpdate(), 10000);
      } else {
        notification.error({ message: 'Start Failed', description: result.message || 'Failed to start online prediction', placement: 'topRight' });
      }
    } catch (e) {
      console.error('Error starting online prediction:', e);
      notification.error({ message: 'Start Failed', description: `Failed to start: ${e.message}`, placement: 'topRight' });
    }
  }

  /**
   * Stop online prediction
   */
  async handleButtonStop() {
    // Do one final fetch before stopping to get latest results
    await this.pollOnlineResults();

    // Immediately stop polling by setting state to false
    this.setState({ isRunningOnline: false });

    // Clear timers immediately
    if (this.statusTimer) {
      clearInterval(this.statusTimer);
      this.statusTimer = null;
    }
    if (this.chartRefreshTimer) {
      clearInterval(this.chartRefreshTimer);
      this.chartRefreshTimer = null;
    }

    try {
      const result = await requestPredictOnlineStop();

      if (result.success) {
        notification.success({ message: 'Stopped', description: 'Online prediction stopped. Final results displayed.', placement: 'topRight' });
      } else {
        notification.error({ message: 'Stop Failed', description: result.message || 'Failed to stop online prediction', placement: 'topRight' });
        // If failed, restore the running state
        this.setState({ isRunningOnline: true });
      }
    } catch (e) {
      console.error('Error stopping online prediction:', e);
      notification.error({ message: 'Stop Failed', description: `Failed to stop: ${e.message}`, placement: 'topRight' });
      // If error, restore the running state
      this.setState({ isRunningOnline: true });
    }
  }

  _relPath(path, base) {
    if (!path) return '-';
    const p = String(path);
    const b = base ? String(base) : null;
    if (b && p.startsWith(b)) return p.slice(b.length).replace(/^\//, '') || '.';
    const idx = p.indexOf('/src/');
    if (idx >= 0) return p.slice(idx + 1);
    const parts = p.split('/').filter(Boolean);
    if (parts.length <= 4) return parts.join('/');
    return '…/' + parts.slice(parts.length - 4).join('/');
  }

  render() {
    const { app, models } = this.props;
    const { mode, modelId, isRunning, predictStats, isRunningOnline, aggregateNormal, aggregateMalicious } = this.state;

    const modelsOptions = getFilteredModelsOptions(app, models);

    const subTitle = 'Offline and online prediction using models';

    let maliciousFlows, predictOutput;
    let normalFlows = 0;
    let totalFlows = 0;
    
    // Handle both offline and online modes
    if (mode === 'online' && (aggregateNormal > 0 || aggregateMalicious > 0)) {
      normalFlows = aggregateNormal;
      maliciousFlows = aggregateMalicious;
      totalFlows = normalFlows + maliciousFlows;
    } else if (mode === 'offline' && predictStats) {
      const predictResult = this.handleTablePredictStats(predictStats);
      maliciousFlows = predictResult.maliciousFlows;
      normalFlows = predictResult.normalFlows;
      totalFlows = normalFlows + maliciousFlows;
    }
    
    if (maliciousFlows > 0) {
      predictOutput = "The model predicts that the given network traffic contains Malicious activity";
    } else if (totalFlows > 0) {
      predictOutput = "The model predicts that the given network traffic is Normal";
    }

    const donutData = [
      { type: 'Normal', value: normalFlows },
      { type: 'Malicious', value: maliciousFlows || 0 },
    ];
    const donutConfig = {
      data: donutData,
      angleField: 'value',
      colorField: 'type',
      radius: 1,
      innerRadius: 0.64,
      legend: { position: 'right' },
      label: {
        type: 'inner',
        offset: '-50%',
        content: ({ percent }) => `${(percent * 100).toFixed(0)}%`,
        style: { fontSize: 14, textAlign: 'center' },
      },
      color: ['#5B8FF9', '#F4664A'],
      interactions: [{ type: 'element-active' }],
      statistic: {
        title: false,
        content: {
          content: totalFlows ? `${totalFlows}` : '',
          style: { fontSize: 16 },
        },
      },
    };

    const maliciousRate = totalFlows > 0 ? maliciousFlows / totalFlows : 0;

    const onSyncPaginate = (pagination) => {
      this.setState({ attackPagination: { current: pagination.current, pageSize: pagination.pageSize } });
    };

    // Analyze malicious flows for top sources
    const analyzeTopSources = () => {
      const { attackRows } = this.state;
      if (!attackRows || attackRows.length === 0) return null;

      const srcIpCounts = {};
      const dstIpCounts = {};

      console.log('[Top Sources] Analyzing', attackRows.length, 'malicious flows');
      
      // Log first row to see structure
      if (attackRows.length > 0) {
        console.log('[Top Sources] First row columns:', Object.keys(attackRows[0]));
        console.log('[Top Sources] First row sample:', attackRows[0]);
      }
      
      attackRows.forEach((row, idx) => {
        // Extract IPs from ip.pkts_per_flow column which has format: "['ip1', 'ip2']"
        const pktsPerFlow = row['ip.pkts_per_flow'];
        let srcIp = null;
        let dstIp = null;
        
        if (pktsPerFlow && typeof pktsPerFlow === 'string') {
          // Parse the Python list string format: "['222.25.140.255', '222.25.140.74']"
          const matches = pktsPerFlow.match(/\['([^']+)',\s*'([^']+)'\]/);
          if (matches && matches.length === 3) {
            srcIp = matches[1];
            dstIp = matches[2];
          }
        }
        
        // Log first few extractions
        if (idx < 3) {
          console.log(`[Top Sources] Row ${idx}:`, { srcIp, dstIp });
        }
        
        // Count source IPs
        if (srcIp && this.isValidIPv4(srcIp)) {
          srcIpCounts[srcIp] = (srcIpCounts[srcIp] || 0) + 1;
        }
        
        // Count destination IPs
        if (dstIp && this.isValidIPv4(dstIp)) {
          dstIpCounts[dstIp] = (dstIpCounts[dstIp] || 0) + 1;
        }
      });

      console.log('[Top Sources] Found:', {
        srcIPs: Object.keys(srcIpCounts).length,
        dstIPs: Object.keys(dstIpCounts).length
      });
      console.log('[Top Sources] srcIpCounts:', srcIpCounts);

      // Convert to sorted arrays (top 10)
      const topSrcIPs = Object.entries(srcIpCounts)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([ip, count]) => ({ name: ip, value: count }));

      // Use destination IPs if no source IPs found
      const topDstIPs = topSrcIPs.length === 0 
        ? Object.entries(dstIpCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10)
            .map(([ip, count]) => ({ name: ip, value: count }))
        : [];

      const result = { 
        topSrcIPs: topSrcIPs.length > 0 ? topSrcIPs : topDstIPs
      };
      
      console.log('[Top Sources] Final result:', result);
      return result;
    };

    const topSourcesData = this.state.attackRows && this.state.attackRows.length > 0 ? analyzeTopSources() : null;
    
    console.log('[Render] attackRows count:', this.state.attackRows ? this.state.attackRows.length : 0);
    console.log('[Render] topSourcesData:', topSourcesData);
    console.log('[Render] State check:', {
      mode,
      predictStats,
      modelId,
      testingDataset: this.state.testingDataset,
      testingPcapFile: this.state.testingPcapFile,
      shouldShowResults: ((mode === 'offline' && predictStats && modelId && this.state.testingPcapFile))
    });

    return (
      <LayoutPage pageTitle="Anomaly Prediction" pageSubTitle={subTitle}>
        
        <Divider orientation="left">
          <h2 style={{ fontSize: '20px' }}>Configuration</h2>
        </Divider>
        
        <Card style={{ marginBottom: 16 }}>
          <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
            <Col flex="none">
              <strong style={{ marginRight: 8 }}>Mode:</strong>
            </Col>
            <Col flex="none">
              <Select
                value={mode}
                onChange={(value) => {
                  // Prevent switching to online if user doesn't have permission
                  if (value === 'online' && !this.props.canPerformOnlineActions) {
                    notification.warning({ message: 'Access Denied', description: 'Administrator privileges required for online predictions', placement: 'topRight' });
                    return;
                  }
                  this.setState({
                    mode: value,
                    modelId: null,
                    interface: null,
                    predictStats: null,
                    attackRows: [],
                    attackFlowColumns: [],
                    mitigationColumns: [],
                    attackCsv: null,
                    aggregateNormal: 0,
                    aggregateMalicious: 0,
                    hasResultsShown: false,
                  });
                }}
                style={{ width: 200 }}
                disabled={isRunning || isRunningOnline}
              >
                <Select.Option value="offline">Offline (PCAP)</Select.Option>
                <Select.Option value="online" disabled={!this.props.canPerformOnlineActions}>
                  <Tooltip title={!this.props.canPerformOnlineActions ? "Admin access required" : ""}>
                    Online (Interface) {!this.props.canPerformOnlineActions && <LockOutlined />}
                  </Tooltip>
                </Select.Option>
              </Select>
            </Col>
          </Row>
          
          <Divider style={{ margin: '16px 0' }} />
        
        {mode === 'offline' ? (
          <>
            <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
              <Col span={6} style={{ textAlign: 'right', paddingRight: 16 }}>
                <strong><span style={{ color: 'red' }}>* </span>Model:</strong>
              </Col>
              <Col span={18}>
                <Select placeholder="Select a model ..."
                  style={{ width: '100%', maxWidth: 500 }}
                  allowClear showSearch
                  value={this.state.modelId}
                  disabled={isRunning}
                  onChange={(value) => {
                    // Clear prediction results when model is cleared
                    if (!value) {
                      this.setState({ 
                        modelId: value, 
                        predictStats: null,
                        attackRows: [],
                        attackFlowColumns: [],
                        mitigationColumns: [],
                        attackCsv: null
                      });
                    } else {
                      this.setState({ modelId: value, predictStats: null });
                    }
                    console.log(`Select model ${value}`);
                  }}
                  options={modelsOptions}
                />
              </Col>
            </Row>
            
            <Row gutter={16} align="top" style={{ marginBottom: 16 }}>
              <Col span={6} style={{ textAlign: 'right', paddingRight: 16 }}>
                <strong><span style={{ color: 'red' }}>* </span>PCAP File:</strong>
              </Col>
              <Col span={18}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                  {this.state.wasUploaded ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <Tag color="green" style={{ margin: 0, padding: '4px 12px', fontSize: '14px' }}>
                        {this.state.testingPcapFile}
                      </Tag>
                      <Button 
                        size="small"
                        disabled={isRunning}
                        onClick={() => {
                          this.setState({
                            testingPcapFile: null,
                            testingDataset: null,
                            wasUploaded: false,
                            predictStats: null,
                            attackRows: [],
                            attackFlowColumns: [],
                            mitigationColumns: [],
                            attackCsv: null
                          });
                        }}
                      >
                        Clear Upload
                      </Button>
                    </div>
                  ) : (
                    <Select
                      placeholder="Select a PCAP file..."
                      value={this.state.testingPcapFile}
                      disabled={isRunning}
                      onChange={(value) => {
                        // Clear all results when pcap is cleared
                        if (!value) {
                          this.setState({ 
                            testingPcapFile: null,
                            testingDataset: null,
                            wasUploaded: false,
                            predictStats: null,
                            attackRows: [],
                            attackFlowColumns: [],
                            mitigationColumns: [],
                            attackCsv: null
                          });
                        } else {
                          // Clear cached report mapping for this PCAP to force fresh analysis
                          try {
                            const raw = localStorage.getItem('pcapToReport');
                            let map = raw ? JSON.parse(raw) : {};
                            if (map && map[value]) {
                              delete map[value];
                              localStorage.setItem('pcapToReport', JSON.stringify(map));
                            }
                          } catch (e) {
                            // ignore
                          }
                          this.setState({ 
                            testingPcapFile: value, 
                            testingDataset: null, 
                            wasUploaded: false, 
                            predictStats: null,
                            attackRows: [],
                            attackFlowColumns: [],
                            mitigationColumns: [],
                            attackCsv: null
                          });
                        }
                      }}
                      showSearch allowClear
                      style={{ width: 280 }}
                    >
                      {(() => {
                        const entries = Array.isArray(this.state.pcapFiles) ? this.state.pcapFiles : [];
                        const byName = new Map();
                        for (const item of entries) {
                          const f = typeof item === 'string' ? { name: item, type: 'sample', path: 'samples' } : item;
                          const existing = byName.get(f.name);
                          if (!existing || (existing.type !== 'user' && f.type === 'user')) {
                            byName.set(f.name, f);
                          }
                        }
                        const users = [];
                        const samples = [];
                        for (const f of byName.values()) {
                          if (f.type === 'user') users.push(f); else samples.push(f);
                        }
                        users.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
                        samples.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
                        const renderOption = (f) => (
                          <Select.Option key={f.name} value={f.name}>
                            <span>
                              {f.type === 'user' && <UserOutlined style={{ marginRight: 8, color: '#1890ff' }} />}
                              {f.type === 'sample' && <DatabaseOutlined style={{ marginRight: 8, color: '#52c41a' }} />}
                              {f.name}
                            </span>
                          </Select.Option>
                        );
                        return [
                          ...users.map(renderOption),
                          ...samples.map(renderOption),
                        ];
                      })()}
                    </Select>
                  )}
                  {this.props.isSignedIn && (
                    <Tooltip title={`Upload your own PCAP file (max ${MAX_PCAP_SIZE_MB} MB)`} placement="top">
                      <Upload
                        beforeUpload={this.beforeUploadPcap}
                        action={`${SERVER_URL}/api/pcaps`}
                        onChange={(info) => this.handleUploadPcap(info)}
                        customRequest={this.processUploadPcap}
                        showUploadList={false}
                        disabled={isRunning}
                      >
                        <Button icon={<UploadOutlined />} disabled={!!this.state.testingPcapFile || isRunning} style={{ width: 212 }}>
                          Upload PCAP
                        </Button>
                      </Upload>
                    </Tooltip>
                  )}
                  {!this.props.isSignedIn && (
                    <Tooltip title="Sign in required">
                      <Button icon={<LockOutlined />} disabled style={{ width: 212 }}>
                        Upload PCAP
                      </Button>
                    </Tooltip>
                  )}
                </div>
              </Col>
            </Row>
            
            <Row gutter={16} align="middle" style={{ marginTop: 24 }}>
              <Col span={6}></Col>
              <Col span={18}>
                <Button type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={this.handlePredictOffline}
                  disabled={ isRunning || !this.state.modelId || !this.state.testingPcapFile }
                  loading={isRunning}
                >
                  Predict
                </Button>
              </Col>
            </Row>
          </>
        ) : (
          <>
            <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
              <Col span={6} style={{ textAlign: 'right', paddingRight: 16 }}>
                <strong><span style={{ color: 'red' }}>* </span>Model:</strong>
              </Col>
              <Col span={18}>
                <Select placeholder="Select a model ..."
                  style={{ width: '100%', maxWidth: 500 }}
                  allowClear showSearch
                  value={this.state.modelId}
                  disabled={isModelIdPresent || isRunningOnline}
                  onChange={(value) => {
                    // Clear prediction results when model is cleared in online mode
                    if (!value) {
                      this.setState({ 
                        modelId: value,
                        predictStats: null,
                        attackRows: [],
                        attackFlowColumns: [],
                        mitigationColumns: [],
                        attackCsv: null,
                        aggregateNormal: 0,
                        aggregateMalicious: 0,
                        hasResultsShown: false,
                      });
                    } else {
                      this.setState({ modelId: value });
                    }
                    console.log(`Select model ${value}`);
                  }}
                  options={modelsOptions}
                />
              </Col>
            </Row>
            
            <Row gutter={16} align="middle" style={{ marginBottom: 16 }}>
              <Col span={6} style={{ textAlign: 'right', paddingRight: 16 }}>
                <strong><span style={{ color: 'red' }}>* </span>Interface:</strong>
              </Col>
              <Col span={18}>
                <Select placeholder="Select a network interface ..."
                  style={{ width: '100%', maxWidth: 500 }}
                  allowClear showSearch
                  disabled={isRunningOnline}
                  value={this.state.interface}
                  onChange={(v) => {
                    // Clear prediction results when interface is cleared in online mode
                    if (!v) {
                      this.setState({ 
                        interface: v,
                        predictStats: null,
                        attackRows: [],
                        attackFlowColumns: [],
                        mitigationColumns: [],
                        attackCsv: null,
                        aggregateNormal: 0,
                        aggregateMalicious: 0,
                        hasResultsShown: false,
                      });
                    } else {
                      this.setState({ interface: v });
                    }
                  }}
                  options={this.state.interfacesOptions}
                />
              </Col>
            </Row>

            <Row gutter={16} align="middle" style={{ marginTop: 24 }}>
              <Col span={6}></Col>
              <Col span={18}>
                <Space size="small">
                  <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={this.handleButtonStart}
                    disabled={ isRunningOnline || !this.state.modelId || !this.state.interface }
                    loading={isRunningOnline}
                  >
                    Start
                  </Button>
                  <Button
                    icon={<StopOutlined />}
                    onClick={this.handleButtonStop}
                    disabled={!isRunningOnline}
                  >
                    Stop
                  </Button>
                </Space>
              </Col>
            </Row>
          </>
        )}
        </Card>

        <Divider orientation="left">
          <h2 style={{ fontSize: '20px' }}>Prediction Results</h2>
        </Divider>
        
        { ((mode === 'offline' && predictStats && modelId && this.state.testingPcapFile) || (mode === 'online' && (this.state.hasResultsShown || aggregateNormal > 0 || aggregateMalicious > 0 || (this.state.attackRows && this.state.attackRows.length > 0)))) ? (
          <>
            {/* Flow Statistics - DPI Style */}
            <Card style={{ marginBottom: 24 }}>
              <div style={{ textAlign: 'center', marginBottom: 16 }}>
                <strong style={{ fontSize: 16 }}>Prediction Summary</strong>
              </div>
              <Row gutter={8}>
                <Col xs={24} sm={8} md={4}>
                  <Card hoverable size="small" style={{ textAlign: 'center', backgroundColor: '#fff', minHeight: '92px' }}>
                    <Statistic
                      title={mode === 'offline' ? 'PCAP File' : 'Interface'}
                      value={mode === 'offline' ? (this.state.testingPcapFile || 'N/A') : (this.state.interface || 'N/A')}
                      valueStyle={{ fontSize: mode === 'online' ? 20 : 11, fontWeight: 'bold', color: '#722ed1', wordBreak: 'break-word', whiteSpace: 'normal', lineHeight: '1.3' }}
                      prefix={<FileTextOutlined style={{ color: '#722ed1', fontSize: '14px' }} />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={8} md={5}>
                  <Card hoverable size="small" style={{ textAlign: 'center', backgroundColor: '#fff' }}>
                    <Statistic
                      title="Total Flows"
                      value={totalFlows}
                      valueStyle={{ fontSize: 20, fontWeight: 'bold', color: '#1890ff' }}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={8} md={5}>
                  <Card hoverable size="small" style={{ textAlign: 'center', backgroundColor: '#fff' }}>
                    <Statistic
                      title="Normal Flows"
                      value={normalFlows}
                      valueStyle={{ fontSize: 20, fontWeight: 'bold', color: '#52c41a' }}
                      prefix={<CheckCircleOutlined />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={8} md={5}>
                  <Card hoverable size="small" style={{ textAlign: 'center', backgroundColor: '#fff' }}>
                    <Statistic
                      title="Malicious Flows"
                      value={maliciousFlows}
                      valueStyle={{ fontSize: 20, fontWeight: 'bold', color: maliciousFlows > 0 ? '#ff4d4f' : '#52c41a' }}
                      prefix={<WarningOutlined />}
                    />
                  </Card>
                </Col>
                <Col xs={24} sm={8} md={5}>
                  <Card hoverable size="small" style={{ textAlign: 'center', backgroundColor: '#fff' }}>
                    <Statistic
                      title="Malicious Rate"
                      value={(maliciousRate * 100).toFixed(1)}
                      suffix="%"
                      valueStyle={{ fontSize: 20, fontWeight: 'bold', color: maliciousRate > 0 ? '#ff4d4f' : '#52c41a' }}
                    />
                  </Card>
                </Col>
              </Row>
              
              {/* Prediction Status Banner - Centered below the boxes, inside card */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, marginTop: 24 }}>
                <span style={{ fontSize: 16, fontWeight: 'bold' }}>{predictOutput}</span>
                {maliciousFlows > 0 ? (
                  <Tag color="error" icon={<WarningOutlined />}>MALICIOUS DETECTED</Tag>
                ) : (
                  <Tag color="success" icon={<CheckCircleOutlined />}>NORMAL TRAFFIC</Tag>
                )}
              </div>
            </Card>
            
            {/* Visualizations */}
            <Row gutter={16} style={{ marginBottom: 24 }}>
              <Col xs={24} lg={12}>
                <Card style={{ height: '100%' }}>
                  <div style={{ marginBottom: 16 }}>
                    <h3 style={{ fontSize: '16px', marginBottom: 4, fontWeight: 600 }}>Flow Distribution</h3>
                    <span style={{ fontSize: '13px', color: '#8c8c8c' }}>
                      Proportion of normal vs malicious flows detected by the model
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300 }}>
                    <Pie {...donutConfig} style={{ height: 280 }} />
                  </div>
                </Card>
              </Col>
              <Col xs={24} lg={12}>
                <Card style={{ height: '100%' }}>
                  <div style={{ marginBottom: 16 }}>
                    <h3 style={{ fontSize: '16px', marginBottom: 4, fontWeight: 600 }}>Top IP Addresses</h3>
                    <span style={{ fontSize: '13px', color: '#8c8c8c' }}>
                      IP addresses most frequently involved in malicious flows
                    </span>
                  </div>
                  {topSourcesData && topSourcesData.topSrcIPs && topSourcesData.topSrcIPs.length > 0 ? (
                    <Bar
                      data={topSourcesData.topSrcIPs}
                      xField="value"
                      yField="name"
                      seriesField="name"
                      legend={false}
                      color="#ff4d4f"
                      label={{
                        position: 'right',
                        formatter: (datum) => datum.value,
                      }}
                      height={Math.max(topSourcesData.topSrcIPs.length * 40, 280)}
                    />
                  ) : (
                    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 300, color: '#8c8c8c' }}>
                      No malicious flows detected yet
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
            
            {this.state.attackRows && this.state.attackRows.length > 0 && (
              <>
                <Divider orientation="left">
                  <h2 style={{ fontSize: '20px' }}>Malicious Flows</h2>
                </Divider>
                <Card>
                  <div style={{ marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ fontSize: '16px', fontWeight: 600, margin: 0 }}>Detected Malicious Flows</h3>
                    <Tooltip title={!this.props.userRole?.isAdmin ? "Admin access required" : ""}>
                      <Button
                        icon={!this.props.userRole?.isAdmin ? <LockOutlined /> : <SendOutlined />}
                        onClick={() => handleBulkMitigationAction({ actionKey: 'send-nats-bulk', rows: this.state.attackRows, isValidIPv4: this.isValidIPv4, entityLabel: 'flows', titleOverride: 'Confirm bulk: Send all to NATS' })}
                        disabled={!(this.state.attackRows && this.state.attackRows.length > 0) || !this.props.userRole?.isAdmin}
                      >
                        Send all to NATS
                      </Button>
                    </Tooltip>
                  </div>
                  <Table
                    dataSource={this.state.attackRows}
                    columns={[...this.state.attackFlowColumns, ...this.state.mitigationColumns]}
                    size="small"
                    style={{ width: '100%' }}
                    scroll={{ x: 'max-content' }}
                    pagination={{ ...this.state.attackPagination, showSizeChanger: true, showTotal: (total) => `Total ${total} flows` }}
                    onChange={(pagination) => onSyncPaginate(pagination)}
                  />
                </Card>
              </>
            )}
            
            <Modal
              title="LIME Explanation"
              open={this.state.limeModalVisible}
              onCancel={() => this.setState({ limeModalVisible: false })}
              footer={<Button onClick={() => this.setState({ limeModalVisible: false })}>Close</Button>}
              width={700}
            >
              <Table
                dataSource={(this.state.limeValues || []).map((row, idx) => ({ key: idx + 1, ...row }))}
                columns={[
                  { title: 'Feature', dataIndex: 'feature' },
                  { title: 'Value', dataIndex: 'value' },
                ]}
                size="small"
                pagination={{ pageSize: 10 }}
              />
            </Modal>
            <Modal
              title="Assistant Explanation"
              open={this.state.assistantModalVisible}
              onCancel={() => this.setState({ assistantModalVisible: false })}
              footer={<Button onClick={() => this.setState({ assistantModalVisible: false })}>Close</Button>}
              width={800}
            >
              {this.state.assistantLoading ? (
                <div style={{ display: 'flex', justifyContent: 'center', padding: 24 }}>
                  <Spin size="large" />
                </div>
              ) : (
                <>
                  <div className="assistant-markdown" style={{ maxHeight: 500, overflowY: 'auto' }}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {this.state.assistantText || ''}
                    </ReactMarkdown>
                  </div>
                  {this.state.assistantTokenInfo && (
                    <div style={{ marginTop: 16, padding: '12px', backgroundColor: '#f6f8fa', borderRadius: '4px' }}>
                      <Typography.Text type="secondary" style={{ fontSize: '12px' }}>
                        <strong>Token Usage:</strong> {this.state.assistantTokenInfo.thisRequest} tokens used this request
                        {this.state.assistantTokenInfo.limit !== Infinity && this.state.assistantTokenInfo.limit != null && (
                          <> - <strong>Total:</strong> {(this.state.assistantTokenInfo.totalUsed || 0).toLocaleString()}/{this.state.assistantTokenInfo.limit.toLocaleString()} 
                          ({this.state.assistantTokenInfo.percentUsed}% used) - <strong>Remaining:</strong> {(this.state.assistantTokenInfo.remaining || 0).toLocaleString()} tokens</>
                        )}
                        {(this.state.assistantTokenInfo.limit === Infinity || this.state.assistantTokenInfo.limit == null) && <> - <strong>Unlimited</strong> (Admin)</>}
                      </Typography.Text>
                    </div>
                  )}
                </>
              )}
            </Modal>
          </>
        ) : null}

      </LayoutPage>
    );
  }
}

const mapPropsToStates = ({ app, models, mmtStatus, reports, predictStatus }) => ({
  app, models, mmtStatus, reports, predictStatus,
});

const mapDispatchToProps = (dispatch) => ({
  fetchApp: () => dispatch(requestApp()),
  fetchAllModels: () => dispatch(requestAllModels()),
  fetchBuildConfigModel: (modelId) => dispatch(requestBuildConfigModel(modelId)),
  fetchAllReports: () => dispatch(requestAllReports()),
  fetchPredictStatus: () => dispatch(requestPredictStatus()),
  fetchRunLime: (modelId, sampleId, numberFeatures) =>
    dispatch(requestRunLime({ modelId, sampleId, numberFeatures })),
});

// Wrap with role check
const PredictPageWithRole = (props) => {
  const userRole = useUserRole();
  return <PredictPage {...props} userRole={userRole} canPerformOnlineActions={userRole.canPerformOnlineActions} isSignedIn={userRole.isSignedIn} isAuthLoaded={userRole.isLoaded} />;
};

export default connect(mapPropsToStates, mapDispatchToProps)(PredictPageWithRole);