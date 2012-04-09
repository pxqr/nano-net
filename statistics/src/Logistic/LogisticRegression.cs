using Nanon.Math.Activator;
using Nanon.Math.Linear;
using Nanon.Model;
using Nanon.Model.Regression;

namespace Nanon.Statistics.Logistic
{
	public class LogisticRegression //: IHypothesis<Vector, double>, IRegression<Vector, double>
	{
		IActivator activator;
		Vector weights;
		
		public LogisticRegression (int inputSize, IActivator activationFunction)
		{
			activator = activationFunction;
		}

		#region IRegression[Vector,System.Double] implementation
		
		public double Predict (Vector input)
		{
			return activator.Activate(weights * input);			
		}
		
		#endregion
		
		#region IHypothesis[Vector,System.Double] implementation
		
		public double Cost (Vector input, double output)
		{
			throw new System.NotImplementedException ();
		}

		public Vector[] Gradient (Vector input, double output)
		{
			var cost = Cost(input, output);
			return new Vector[] { cost * input };
		}

		public void Correct(Vector[] gradient)
		{
			weights += gradient[0];
		}

		public Vector[] ZeroGradient {
			get 
			{
				return new Vector[] { weights.ZeroCopy() };
			}
		}
		
		#endregion
	}
}

