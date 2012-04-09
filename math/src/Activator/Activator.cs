
namespace Nanon.Math.Activator
{
	public interface IActivator
	{
		double Activate(double x);		
		double Derivative(double fx);
	}
	
	public class Logistic : IActivator
	{
		public double Activate(double x) 
		{
			return 1 / (1 + System.Math.Exp(-x));
		}
		
		public double Derivative(double fx)
		{
			return fx * (1 - fx);
		}
	}
	
	public class Tanh : IActivator
	{
		public double Activate(double x) 
		{
			return System.Math.Tanh(x);
		}
		
		public double Derivative(double fx)
		{
			return 1 - fx * fx;
		}
	}
}

