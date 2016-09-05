function n_correct = testclassification(trainingfiles, testfile)
%
% testclassification(trainingfiles, testfile)
% 
% Uses the data in *trainingfiles* to build a classifier and tests
% the classifier on the data in *testfile*. *n_correct* contains for each
% number of blocks (1-20) the number of correctly classified items. If no
% output arguments are given *n_correct* is plotted.
% The training files and the test file have to be built with extracttrials.
%
% Example: testclassification({'s2','s3','s4'},'s1')

% Author: Ulrich Hoffmann - EPFL, 2006
% Copyright: Ulrich Hoffmann - EPFL


%% subsets of electrodes
% Fz, Cz, Pz, Oz
% channels = [31 32 13 16];                

% Fz, Cz, Pz, Oz, P7, P3, P4, P8 
channels = [31 32 13 16 11 12 19 20];                  

% Fz, Cz, Pz, Oz, P7, P3, P4, P8, O1, O2, C3, C4, FC1, FC2, CP1, CP2 
% channels = [31 32 13 16 11 12 19 20 15 17 8 23 5 26 9 22];

% All electrodes
% channels = [1:32];


%% load training files and concatenate data and labels into two big arrays
x = [];
y = [];
for i = 1:length(trainingfiles);
    fprintf('loading %s\n',trainingfiles{i});
    f = load(trainingfiles{i});
    n_runs = length(f.runs);
    for j = 1:n_runs;
        x = cat(3,x,f.runs{j}.x);
        y = [y f.runs{j}.y];
    end
end


%% select channels, windsorize, normalize, bayesian lda  
x = x(channels,:,:);
w = windsor;
w = train(w,x,0.1);
x = apply(w,x);
n = normalize;
n = train(n,x,'z-score');
x = apply(n,x);

n_channels = length(channels);
n_samples = size(x,2);
n_trials = size(x,3);
x = reshape(x,n_samples*n_channels,n_trials);

b = bayeslda(1);
b = train(b,x,y);


%% load testfile and do classification
f = load(testfile);
n_runs = length(f.runs);
n_blocks = 20;
n_correct = zeros(1,n_blocks);
for i = 1:n_runs
    x = f.runs{i}.x(channels,:,:);
    x = apply(w,x);
    x = apply(n,x);
    n_trials = size(x,3);
    x = reshape(x,n_channels*n_samples,n_trials);
    y = classify(b,x);
    scores = zeros(1,6);
    for j = 1:n_blocks
        start = (j-1)*6+1;
        stop  = (j)*6;
        stimulussequence = f.runs{i}.stimuli(start:stop);
        scores(stimulussequence) = scores(stimulussequence) + ...
            y(start:stop);
        [dummy,idx] = max(scores);
        if (idx == f.runs{i}.target)
            n_correct(j) = n_correct(j)+1;
        end
    end
end


%% if no output arguments plot the results
if nargout == 0
    plot(n_correct);
    axis([1 20 0 6]);
    xlabel('Number of blocks');
    ylabel('Number of correct classifications');
end