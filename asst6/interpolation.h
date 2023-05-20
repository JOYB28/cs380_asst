#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "rigtform.h"

namespace Interpolation {

	// ...

	// Catmull-Rom interpolation of two RigTForms
	// TODO: compute the Catmull-Rom interpolation of four RigTForm inputs and return the result
	// Note: To Catmull-Rom interpolate two RigTFrom rbt0 and rbt1, we need 4 Keyframes keyframe rbt_1, rbt0, rbt1, rbt2.
	// 		 You can use the lerp and slerp functions you implemented above.
	inline RigTForm CatmullRom(const RigTForm& rbt_1, const RigTForm& rbt0, const RigTForm& rbt1, const RigTForm& rbt2, const double& alpha) {
		return RigTForm();	// Replace this value with your own code
	}

}
#endif