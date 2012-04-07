using System;

using Nanon.Data;
using Nanon.Model;
using Nanon.Math.Linear;

namespace Nanon.Learning.Tools
{
	public class GradientChecker
	{
		IHypothesis<Vector, Vector> hypothesis;
		
		public GradientChecker(IHypothesis<Vector, Vector> hypothesisA)
		{
			hypothesis = hypothesisA;
		}
		
		public Vector FindGrad(Vector input, Vector output)
		{
			var inputSize = input.Size;
			var grad = input.ZeroCopy();
			var diff = input.ZeroCopy();
			var epsilon = 0.0001d;
			
			for (var i = 0; i < inputSize; ++i)
			{
				diff[i] = epsilon;
				var lossDown = hypothesis.Cost(input + diff, output);
				var lossUp   = hypothesis.Cost(input + diff, output);
				grad[i] = (lossUp - lossDown) / (2 * epsilon);
				diff[i] = 0;
			}
			
			return grad;
		}
		
		public double Check(Vector input)
		{
			throw new NotImplementedException();
		}
	}
}

