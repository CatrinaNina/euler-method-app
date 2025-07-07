import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import io

def euler_method(k, x0, y0, x_target, h):
    """
    Implement Euler's method for solving dy/dx = ky
    
    Args:
        k: differential equation parameter
        x0: initial x value
        y0: initial y value
        x_target: target x value to stop at
        h: step size
    
    Returns:
        DataFrame with step-by-step calculations
    """
    # Calculate number of steps
    n_steps = int((x_target - x0) / h)
    
    # Initialize arrays
    x_values = [x0]
    y_values = [y0]
    slopes = []
    
    # Current values
    x_current = x0
    y_current = y0
    
    # Perform Euler's method iterations
    for i in range(n_steps):
        # Calculate slope at current point
        slope = k * y_current
        slopes.append(slope)
        
        # Calculate next point
        x_next = x_current + h
        y_next = y_current + h * slope
        
        # Store values
        x_values.append(x_next)
        y_values.append(y_next)
        
        # Update current values
        x_current = x_next
        y_current = y_next
    
    # Create DataFrame for step-by-step calculations
    steps_data = []
    for i in range(len(slopes)):
        steps_data.append({
            'Step': i + 1,
            'x_i': x_values[i],
            'y_i': y_values[i],
            'dy/dx = ky_i': slopes[i],
            'x_{i+1}': x_values[i + 1],
            'y_{i+1}': y_values[i + 1]
        })
    
    steps_df = pd.DataFrame(steps_data)
    
    # Create solution DataFrame
    solution_df = pd.DataFrame({
        'x': x_values,
        'y_numerical': y_values
    })
    
    return steps_df, solution_df

def analytical_solution(k, x0, y0, x_values):
    """
    Calculate the analytical solution: y = y0 * e^(k*(x-x0))
    
    Args:
        k: differential equation parameter
        x0: initial x value
        y0: initial y value
        x_values: array of x values
    
    Returns:
        array of analytical y values
    """
    return y0 * np.exp(k * (np.array(x_values) - x0))

def calculate_error(y_numerical, y_analytical):
    """
    Calculate absolute and relative errors
    """
    abs_error = np.abs(y_numerical - y_analytical)
    rel_error = np.abs((y_numerical - y_analytical) / y_analytical) * 100
    return abs_error, rel_error

def export_results_to_csv(steps_df, solution_df, analytical_df):
    """
    Export results to CSV format
    """
    # Combine all data
    combined_df = solution_df.copy()
    combined_df['y_analytical'] = analytical_df['y_analytical']
    combined_df['absolute_error'] = analytical_df['absolute_error']
    combined_df['relative_error_percent'] = analytical_df['relative_error_percent']
    
    # Convert to CSV
    csv_buffer = io.StringIO()
    combined_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def main():
    st.title("Euler's Method for Differential Equations")
    st.subheader("Solving dy/dx = ky")
    
    # Create sidebar for inputs
    st.sidebar.header("Input Parameters")
    
    # Input fields
    k = st.sidebar.number_input(
        "Differential equation parameter (k)",
        value=0.5,
        step=0.1,
        format="%.3f",
        help="The constant k in the equation dy/dx = ky"
    )
    
    x0 = st.sidebar.number_input(
        "Initial x value (x₀)",
        value=0.0,
        step=0.1,
        format="%.3f",
        help="Starting x coordinate"
    )
    
    y0 = st.sidebar.number_input(
        "Initial y value (y₀)",
        value=1.0,
        step=0.1,
        format="%.3f",
        help="Starting y coordinate"
    )
    
    x_target = st.sidebar.number_input(
        "Target x value",
        value=2.0,
        step=0.1,
        format="%.3f",
        help="X value to stop the calculation at"
    )
    
    h = st.sidebar.number_input(
        "Step size (h)",
        value=0.1,
        min_value=0.001,
        max_value=1.0,
        step=0.01,
        format="%.3f",
        help="Step size for Euler's method"
    )
    
    # Error handling
    if h <= 0:
        st.error("Step size must be positive!")
        return
    
    if x_target <= x0:
        st.error("Target x value must be greater than initial x value!")
        return
    
    if abs(k) > 10:
        st.warning("Large k values may lead to numerical instability!")
    
    # Calculate solutions
    try:
        steps_df, solution_df = euler_method(k, x0, y0, x_target, h)
        
        # Calculate analytical solution
        y_analytical = analytical_solution(k, x0, y0, solution_df['x'])
        
        # Calculate errors
        abs_error, rel_error = calculate_error(solution_df['y_numerical'], y_analytical)
        
        # Create analytical DataFrame
        analytical_df = pd.DataFrame({
            'x': solution_df['x'],
            'y_analytical': y_analytical,
            'absolute_error': abs_error,
            'relative_error_percent': rel_error
        })
        
        # Display differential equation
        st.write("### Differential Equation")
        st.latex(r"\frac{dy}{dx} = ky")
        st.write(f"where k = {k}")
        
        # Display initial conditions
        st.write("### Initial Conditions")
        st.write(f"Initial point: ({x0}, {y0})")
        st.write(f"Target x: {x_target}")
        st.write(f"Step size: {h}")
        st.write(f"Number of steps: {len(steps_df)}")
        
        # Display analytical solution
        st.write("### Analytical Solution")
        st.latex(r"y = y_0 \cdot e^{k(x-x_0)}")
        st.write(f"y = {y0} × e^({k}(x-{x0}))")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Plot", "📋 Step-by-Step", "📈 Comparison", "📁 Export"])
        
        with tab1:
            st.write("### Solution Plot")
            
            fig = go.Figure()
            
            # Add numerical solution
            fig.add_trace(go.Scatter(
                x=solution_df['x'],
                y=solution_df['y_numerical'],
                mode='lines+markers',
                name='Numerical (Euler)',
                line=dict(color='blue'),
                marker=dict(size=6)
            ))
            
            # Add analytical solution
            fig.add_trace(go.Scatter(
                x=analytical_df['x'],
                y=analytical_df['y_analytical'],
                mode='lines',
                name='Analytical',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"Euler's Method Solution (k={k}, h={h})",
                xaxis_title="x",
                yaxis_title="y",
                hovermode='x unified',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.write("### Step-by-Step Calculations")
            st.write("Each step shows: x_i, y_i, slope = ky_i, x_{i+1}, y_{i+1} = y_i + h×slope")
            
            # Format the DataFrame for better display
            display_df = steps_df.copy()
            for col in display_df.columns:
                if col != 'Step':
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}")
            
            st.dataframe(display_df, use_container_width=True)
        
        with tab3:
            st.write("### Numerical vs Analytical Comparison")
            
            # Create comparison table
            comparison_df = pd.DataFrame({
                'x': solution_df['x'],
                'Numerical': solution_df['y_numerical'],
                'Analytical': analytical_df['y_analytical'],
                'Absolute Error': abs_error,
                'Relative Error (%)': rel_error
            })
            
            # Format for display
            display_comparison = comparison_df.copy()
            for col in ['Numerical', 'Analytical', 'Absolute Error']:
                display_comparison[col] = display_comparison[col].apply(lambda x: f"{x:.6f}")
            display_comparison['Relative Error (%)'] = display_comparison['Relative Error (%)'].apply(lambda x: f"{x:.4f}%")
            
            st.dataframe(display_comparison, use_container_width=True)
            
            # Error analysis
            st.write("### Error Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Maximum Absolute Error", f"{np.max(abs_error):.6f}")
                st.metric("Mean Absolute Error", f"{np.mean(abs_error):.6f}")
            
            with col2:
                st.metric("Maximum Relative Error", f"{np.max(rel_error):.4f}%")
                st.metric("Mean Relative Error", f"{np.mean(rel_error):.4f}%")
            
            # Error plot
            fig_error = go.Figure()
            fig_error.add_trace(go.Scatter(
                x=solution_df['x'],
                y=abs_error,
                mode='lines+markers',
                name='Absolute Error',
                line=dict(color='orange')
            ))
            
            fig_error.update_layout(
                title="Absolute Error vs x",
                xaxis_title="x",
                yaxis_title="Absolute Error",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_error, use_container_width=True)
        
        with tab4:
            st.write("### Export Results")
            
            # Generate CSV data
            csv_data = export_results_to_csv(steps_df, solution_df, analytical_df)
            
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=f"euler_method_results_k{k}_h{h}.csv",
                mime="text/csv"
            )
            
            # Display summary
            st.write("### Summary")
            st.write(f"- **Differential Equation**: dy/dx = {k}y")
            st.write(f"- **Initial Conditions**: ({x0}, {y0})")
            st.write(f"- **Integration Range**: x ∈ [{x0}, {x_target}]")
            st.write(f"- **Step Size**: {h}")
            st.write(f"- **Number of Steps**: {len(steps_df)}")
            st.write(f"- **Final Numerical Value**: {solution_df['y_numerical'].iloc[-1]:.6f}")
            st.write(f"- **Final Analytical Value**: {analytical_df['y_analytical'].iloc[-1]:.6f}")
            st.write(f"- **Final Absolute Error**: {abs_error[-1]:.6f}")
            st.write(f"- **Final Relative Error**: {rel_error[-1]:.4f}%")
    
    except Exception as e:
        st.error(f"An error occurred during calculation: {str(e)}")
        st.write("Please check your input parameters and try again.")

if __name__ == "__main__":
    main()
