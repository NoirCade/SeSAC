import open3d as o3d
import numpy as np
import plotly.graph_objects as go


def show_point_cloud(points, color_axis, width_size=1500, height_size=800, coordinate_frame=True):
    '''
    points : (N, 3) size of ndarray
    color_axis : 0, 1, 2
    '''
    assert points.shape[1] == 3
    assert color_axis==0 or color_axis==1 or color_axis==2   
    
    
    # Create a scatter3d Plotly plot
    plotly_fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color=points[:, color_axis], # Set color based on Z-values
            colorscale='jet', # Choose a color scale
            colorbar=dict(title='value') # Add a color bar with a title
        )
    )])

    x_range = points[:, 0].max()*0.9 - points[:, 0].min()*0.9
    y_range = points[:, 1].max()*0.9 - points[:, 1].min()*0.9
    z_range = points[:, 2].max()*0.9 - points[:, 2].min()*0.9

    # Adjust the Z-axis scale
    plotly_fig.update_layout(
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=x_range, y=y_range, z=z_range), # Here you can set the scale of the Z-axis     
        ),
        width=width_size, # Width of the figure in pixels
        height=height_size, # Height of the figure in pixels
        showlegend=False
    )
    
    if coordinate_frame:
        # Length of the axes
        axis_length = 1

        # Create lines for the axes
        lines = [
            go.Scatter3d(x=[0, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red')),
            go.Scatter3d(x=[0, 0], y=[0, axis_length], z=[0, 0], mode='lines', line=dict(color='green')),
            go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_length], mode='lines', line=dict(color='blue'))
        ]

        # Create cones (arrows) for the axes
        cones = [
            go.Cone(x=[axis_length], y=[0], z=[0], u=[axis_length], v=[0], w=[0], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False),
            go.Cone(x=[0], y=[axis_length], z=[0], u=[0], v=[axis_length], w=[0], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False),
            go.Cone(x=[0], y=[0], z=[axis_length], u=[0], v=[0], w=[axis_length], sizemode='absolute', sizeref=0.1, anchor='tail', showscale=False)
        ]

        # Add lines and cones to the figure
        for line in lines:
            plotly_fig.add_trace(line)
        for cone in cones:
            plotly_fig.add_trace(cone)

    # Show the plot
    plotly_fig.show()


if __name__ == '__main__':
    pcd_path = "./1K_A15_O032_clear_day_summer_02000641.pcd"
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)
    show_point_cloud(points, 2)