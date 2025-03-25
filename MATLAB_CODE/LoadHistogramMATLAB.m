width = 647;
height = 24;


original_path = 'C:\Users\ruber\Downloads\download.raw';
edge_map_path = 'C:\Users\ruber\Downloads\edge_map.raw';

fid = fopen(original_path, 'rb');
raw_img = fread(fid, width * height, 'uint8=>uint8');
fclose(fid);
original_img = reshape(raw_img, [width, height])';


fid = fopen(edge_map_path, 'rb');
edge_data = fread(fid, width * height, 'uint8=>uint8');
fclose(fid);
edge_map = reshape(edge_data, [width, height])';

% --- Convert original grayscale image to RGB ---
original_rgb = repmat(original_img, [1, 1, 3]);  

% --- Create red overlay for edges ---
overlay_img = original_rgb;
edge_mask = edge_map > 0;  % Binary mask where edges are detected

% Set red channel to max where edges are found
overlay_img(:,:,1) = uint8(double(overlay_img(:,:,1)) + 255 * double(edge_mask));
overlay_img(:,:,2) = uint8(double(overlay_img(:,:,2)) .* ~double(edge_mask));
overlay_img(:,:,3) = uint8(double(overlay_img(:,:,3)) .* ~double(edge_mask));

% --- Display the result ---
figure;
imshow(overlay_img);
title('Overlay of Detected Edges on Original Image (in Red)');



