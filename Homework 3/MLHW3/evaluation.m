function [SSE,RMSE,RSE,R_sq] = evaluation(VS_result, pre_result)
% Sum of square error(SSR)
SSE = sum((VS_result-pre_result).^2);
% Root mean square error (RMSE)
RMSE = sqrt(SSE/length(VS_result));
% Relative square error(RSE)
r_avg = mean(VS_result);
SSR = sum((pre_result-r_avg).^2);
SST = sum((r_avg-VS_result).^2);
RSE = SSE/SST;
% Coefficient of determination
R_sq = SSR/SST;
end