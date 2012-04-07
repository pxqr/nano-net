using System;

using Nanon.Math.Sigmoid;

namespace Nanon.Math.Activator
{	
	public class Tanh : IActivator
	{
		public double Activate(double x) 
		{
			return Sigmoids.Tanh(x);
		}
		
		public double Derivative(double x, double fx)
		{
			return Sigmoids.DTanhX(fx);
		}
	}
}