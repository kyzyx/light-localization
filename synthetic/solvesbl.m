#!/usr/bin/octave -qf

arg_list = argv();
addpath(genpath('../SB2_Release_200'));
run(arg_list{1});
Y = sum(A,2);
run(arg_list{2});
OPTIONS = SB2_UserOptions('FixedNoise', true);
r = [1 0.5 0.1 0.05 0.01 0.005 0.001 0.0005];
r = fliplr(r);
X = zeros(size(A,2),length(r));
for i=1:length(r)
    SETTINGS = SB2_ParameterSettings('NoiseStd', r(i));
    [p, h] = SparseBayes("Gaussian", A, Y,OPTIONS,SETTINGS);
    X(p.Relevant,i) = p.Value;
end
dlmwrite(arg_list{3}, X, 'delimiter', ' ', 'precision', '%.8f');
