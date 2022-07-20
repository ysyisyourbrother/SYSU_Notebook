function new_img = maxmin( img ) 
    [height, width] = size(img);
    max_pixel = max(max(img));
    min_pixel = min(min(img));
    for i = 1 : height
        for j = 1 : width
            new_img(i, j) = 255 * (img(i, j) - min_pixel) / (max_pixel - min_pixel);
        end
    end
    new_img = uint8(new_img);
end