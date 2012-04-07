using Nanon.Math.Sigmoid;

namespace Nanon.Math.Activator
{
	public class Logistic : IActivator
	{
		public double Activate(double x) 
		{
			return Sigmoids.Logistic(x);
		}
		
		public double Derivative(double x, double fx)
		{
			return Sigmoids.DLogisticX(fx);
		}
	}
}

