% Script for loading the dataset

% Clear console
clear; clc; close all;

% Set path
sOldPath    = path;
sDataPath   = './DATA/';
addpath('Utils');

% Load dataset
load(sprintf('%sDataStockRetPrediction.mat',sDataPath))

% The data covers:
%   eom:                Date of characteristocs (end-of-month)
%   id:                 JKP id of asset
%   size_grp:           Size group of asset  
%   ret_exc_lead1m:     Return in time t+1
%   [other columns]:    Characteristics in time t

% Get unique dates
dtDates = unique(tData.eom);

% Get factor names
vAssetID = unique(tData.id);

% Get returns (time t+1) in wide format
vRetWide = tData.ret_exc_lead1m;

% Get factor returns in T x N format
mRet = fPanel2Matrix(vRetWide, tData.eom, tData.id, dtDates, vAssetID);

% Get characteristics in wide format (time t)
mCharsWide = tData{:,5:end};
cCharNames = tData.Properties.VariableNames(5:end);

% Transform characteristics into cross-sectional ranks and map
% to [-0.5,0.5] interval
mCharsWide = grouptransform(mCharsWide, tData.eom, @(x)fCrossSectTransChars(x,'rank'));

% Get factor characteristics in T x N x M format
mChars = fPanel2Matrix(mCharsWide, tData.eom, tData.id, dtDates, vAssetID);

% Restore path
path(sOldPath);