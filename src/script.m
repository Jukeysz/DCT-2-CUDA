% ---------------------------
% Parameters
% ---------------------------
folder = 'dataset';                     % your folder of .ppm files
files  = dir(fullfile(folder,'*.ppm'));
if isempty(files)
    error('No .ppm files found in "%s".', folder);
end

% 1) Sort filenames lexicographically (same as std::sort on paths)
[~, order] = sort({files.name});
files      = files(order);

% Light-field angular dims:
U = 15;    % number of rows (u)
V = 15;    % number of cols (v)
if numel(files) ~= U*V
    warning('Expected %d images, but found %d.', U*V, numel(files));
end

% Read one image to get spatial size:
firstImg = imread(fullfile(folder, files(1).name));
if size(firstImg,3)==3
    firstImg = rgb2gray(firstImg);
end
H = size(firstImg,1);
W = size(firstImg,2);

% ---------------------------
% 2) Load into a 4-D array
%    data(y, x, u, v)
% ---------------------------
data = zeros(H, W, U, V);
for idx = 1:numel(files)
    % replicate: u = idx / V; v = idx % V in 0-based C++
    u = floor((idx-1)/V) + 1;      % MATLAB 1-based
    v = mod(idx-1, V) + 1;

    img = imread(fullfile(folder, files(idx).name));
    if size(img,3)==3
        img = rgb2gray(img);
    end
    data(:,:,u,v) = double(img);
end

% ---------------------------
% 3) Compute separable 4-D DCT-II
% ---------------------------
coeff = data;
for dim = 1:4
    coeff = dct(coeff, [], dim);
end

% ---------------------------
% 4) Extract & display first 4×4×4×4 block
% ---------------------------
block4 = coeff(1:4, 1:4, 1:4, 1:4);
disp('First 4×4×4×4 block of DCT-II coefficients:');
disp(block4);

% ---------------------------
% Note on ordering
% ---------------------------
fprintf([ ...
  '\nNOTE: This uses the same sorted-filenames + (u,v) mapping ' ...
  'as your C++ loader: u = floor((idx-1)/V), v = mod(idx-1,V). ' ...
  'If you change that, the coefficients will shift accordingly.\n' ...
]);
