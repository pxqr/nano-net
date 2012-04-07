using System;
using Nanon.Model;

namespace Nanon.Model.Regression
{
	public interface IRegression<InputT, OutputT>
	{
		OutputT Predict(InputT input);
	}
}

