% Script for preparing the dataset

% Clear console
clear; clc; close all;

% Set path
sOldPath    = path;
sDataPath   = './DATA/';
addpath('Utils');

% =====================================================================
% Predictor variables (JKP Global Factor Data) used to replicate
% Lewellen (2015), "The Cross-Section of Expected Stock Returns"
% -- Model 3 (large/15-variable specification) --
% =====================================================================
%
% id                : Firm identifier
% eom                : End-of-month date
% size_grp:          : Defines the size group of an asset
% ret_exc_lead1m     : Dependent variable - next-month excess return
%
% --- Predictors (JKP mnemonics, closest proxies to Lewellen's variables) ---
%
% market_equity      : Size            -> log market value of equity (t-1)
% be_me              : Value           -> log book-to-market equity (t-1)
% ret_12_1           : Momentum        -> return from month t-12 to t-2
% chcsho_12m         : Share issuance  -> log growth in shares outstanding, 12m
% oaccruals_at       : Accruals        -> operating accruals scaled by assets
% ni_at              : Profitability   -> return on assets (prior fiscal year)
% at_gr1             : Asset growth    -> log growth in total assets
% beta_60m           : Beta            -> market beta from 60m rolling returns
% div12m_me          : Dividend yield  -> trailing 12m dividends / price
% ret_60_12          : Long-term return -> return from month t-60 to t-12
%                       (proxy for Lewellen's 3-year return, t-36 to t-13)
% rvol_252d          : Volatility      -> daily-return volatility, past 252d
% turnover_126d      : Turnover        -> average share turnover, past 126d
% debt_me            : Leverage        -> total debt / market equity
%                       (proxy for Lewellen's market leverage)
% sale_me            : Sales-to-price  -> sales (prior fiscal year) / market equity
%
% Note: JKP definitions are close but not identical to Lewellen's original
% variable construction (e.g. window lengths for issuance, return, turnover,
% and leverage differ slightly). This should be treated as a replication
% "in the spirit of" Lewellen (2015), not an exact 1:1 reproduction.
% =====================================================================

% Define columns to import
cColsImport = {'id','eom','size_grp','ret_exc_lead1m','market_equity','be_me',...
    'ret_12_1','chcsho_12m','oaccruals_at','ni_at','at_gr1','beta_60m',...
    'div12m_me','ret_60_12','rvol_252d','turnover_126d',...
    'debt_me','sale_me'};

% Load data
tData = parquetread([sDataPath,'USA.parquet'],'SelectedVariableNames',cColsImport);

% Drop rows with missing returns or market capitalization
tData(isnan(tData.ret_exc_lead1m)|isnan(tData.market_equity),:) = [];

% Drop all nano and micro stocks
tData(ismember(tData.size_grp,{'nano','micro'}),:) = [];

% Save dataset
save(sprintf('%sDataStockRetPrediction.mat',sDataPath),'tData')

% Restore path
path(sOldPath);