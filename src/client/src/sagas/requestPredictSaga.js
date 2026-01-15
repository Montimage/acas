// watcher saga -> actions -> worker saga
import {
    call,
    put,
    takeEvery,
  } from 'redux-saga/effects';
  
  import {
    requestPredictStatus,
  } from '../api';
  import {
    setNotification,
    setPredictStatus,
  } from '../actions';
  
  
  function* handleRequestPredictStatus() {
    try {
      const status = yield call(() => requestPredictStatus());
      // dispatch data
      yield put(setPredictStatus(status));
    } catch (error) {
      // dispatch error
      yield put(setNotification({ type: "error", message: error }));
    }
  }
  
  function* watchDatasets() {
    yield takeEvery('REQUEST_PREDICT_STATUS', handleRequestPredictStatus);
  }
  
  export default watchDatasets;