using System;
using System.Collections.Generic;

namespace GraphBased
{	
	public class Neuron
	{
		List<Neuron> posterior;  // outputs
		List<double> weights;
		double signal;
		double output;
		double error;
		
		// forward propagation
		void Transfer()
		{
			Emit (output);
		}
		
		// for input neurons only!
		void Emit(double x)
		{
			for (var i = 0; i < weights.Count; ++i)
				posterior[i].signal += weights[i] * x;
		}
		
		// backpropagation
		void Receive()
		{
			var e = 0.0d;
			for (int i = 0; i < weights.Count; ++i)
				e += posterior[i].error * weights[i];
			
			var deriv = 1 - signal * signal;
			error = e * deriv;
		}
		
		// set output
		void Activate()
		{
			output = System.Math.Tanh(signal);
		}
		
		// set error. For output neurons only!
		void Error(double error)
		{
			this.error = error;
		}
		
		void Gradient(double input)
		{
			
		}
	}
}

