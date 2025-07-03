import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Directory with heatmap images
    input_dir = 'output/figures'
    output_dir = 'output/figures'
    os.makedirs(output_dir, exist_ok=True)

    # Correlation metrics and UQ methods
    correlation_types = ['pearson', 'spearman', 'kendall']
    uq_methods = ['dropout_pu', 'softmax_pu']

    # Collect all heatmap paths
    heatmaps = {uq: {} for uq in uq_methods}

    loaded_datasets = set()

    for fname in os.listdir(input_dir):
        if fname.endswith('.png') and fname.startswith('correlation_matrix_'):
            parts = fname.replace('.png', '').split('_')
            if len(parts) < 5:
                continue  # malformed filename
            corr = parts[2]
            dataset = parts[3:-2]
            dataset = '_'.join(dataset)
            uq_method = parts[-2:]
            uq_method = '_'.join(uq_method)
            if uq_method in uq_methods and corr in correlation_types:
                heatmaps[uq_method].setdefault(dataset, {})[corr] = os.path.join(input_dir, fname)
                loaded_datasets.add(dataset)
            else:
                print(f"\033[93mWARNING:\033[0m Skipping {fname} because it is not a valid heatmap.")
    
    print(f"Included heatmaps for {len(loaded_datasets)} datasets:")
    datasets = sorted(loaded_datasets.to_list())
    for dataset in datasets:
        print(f"  {dataset}")


    def create_grid_image(uq_method, grid_data):
        datasets = sorted(grid_data.keys())  # alphabetically sorted
        n_rows = len(datasets)
        n_cols = len(correlation_types)

        # Determine image size using any available image
        sample_img_path = None
        for d in datasets:
            for corr in correlation_types:
                path = grid_data[d].get(corr)
                if path and os.path.isfile(path):
                    sample_img_path = path
                    break
            if sample_img_path:
                break

        if not sample_img_path:
            print(f"No valid images found for UQ method '{uq_method}'. Skipping.")
            return

        sample_image = Image.open(sample_img_path)
        img_w, img_h = sample_image.size

        # Create a blank white image for placeholders
        blank_image = Image.new('RGB', (img_w, img_h), color='white')

        # Header sizes
        label_pad_w = 300  # space for row (dataset) names
        label_pad_h = 100  # space for column (correlation) names

        # Create final grid canvas with label padding
        grid_img = Image.new(
            'RGB',
            (n_cols * img_w + label_pad_w, n_rows * img_h + label_pad_h),
            color='white'
        )
        draw = ImageDraw.Draw(grid_img)

        # Optional: Use larger/bolder font if available
        try:
            font = ImageFont.truetype("arial.ttf", 80)
        except IOError:
            font = ImageFont.load_default()

        # Draw column headers (correlation types)
        for col_idx, corr in enumerate(correlation_types):
            x = label_pad_w + col_idx * img_w + img_w // 2
            y = label_pad_h // 2
            draw.text((x, y), corr.upper(), font=font, fill='black', anchor='mm')

        # Draw row headers (dataset names)
        for row_idx, dataset in enumerate(datasets):
            x = label_pad_w // 2
            y = label_pad_h + row_idx * img_h + img_h // 2
            draw.text((x, y), dataset, font=font, fill='black', anchor='mm')

        # Paste heatmaps into the grid
        for row_idx, dataset in enumerate(datasets):
            for col_idx, corr in enumerate(correlation_types):
                img_path = grid_data[dataset].get(corr)
                if img_path and os.path.isfile(img_path):
                    img = Image.open(img_path)
                else:
                    img = blank_image.copy()
                x = label_pad_w + col_idx * img_w
                y = label_pad_h + row_idx * img_h
                grid_img.paste(img, (x, y))

        out_path = os.path.join(output_dir, f'correlation_overview_all_datasets_{uq_method}_grid.png')
        grid_img.save(out_path)
        print(f"Saved: {out_path}")


    # Function to create a grid image
    def create_grid_image_old(uq_method, grid_data):
        datasets = sorted(grid_data.keys())
        n_rows = len(datasets)
        n_cols = len(correlation_types)

        # Determine image size using any available image
        sample_img_path = None
        for d in datasets:
            for corr in correlation_types:
                path = grid_data[d].get(corr)
                if path and os.path.isfile(path):
                    sample_img_path = path
                    break
            if sample_img_path:
                break

        if not sample_img_path:
            print(f"No valid images found for UQ method '{uq_method}'. Skipping.")
            return

        sample_image = Image.open(sample_img_path)
        img_w, img_h = sample_image.size

        # Create a blank white image for placeholders
        blank_image = Image.new('RGB', (img_w, img_h), color='white')

        # Create the full grid image
        grid_img = Image.new('RGB', (n_cols * img_w, n_rows * img_h), color='white')

        for row_idx, dataset in enumerate(datasets):
            for col_idx, corr in enumerate(correlation_types):
                img_path = grid_data[dataset].get(corr)
                if img_path and os.path.isfile(img_path):
                    img = Image.open(img_path)
                else:
                    img = blank_image.copy()
                grid_img.paste(img, (col_idx * img_w, row_idx * img_h))

        out_path = os.path.join(output_dir, f'{uq_method}_grid.png')
        grid_img.save(out_path)
        print(f"Saved: {out_path}")


    # Create grids for both UQ methods
    for uq in uq_methods:
        create_grid_image(uq, heatmaps[uq])