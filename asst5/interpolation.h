#ifndef INTERPOLATION_H
#define INTERPOLATION_H

#include "rigtform.h"

namespace Interpolation {

	// Linear interpolation of two coordinate vectors
	// TODO: compute the linear interpolation of two Cvec3 inputs and return the result
	inline Cvec3 lerp(const Cvec3& c0, const Cvec3& c1, const double& alpha) {
		return Cvec3();	// Replace this value with your own code
	}

	// Spherical interpolation of two quaternions
	// TODO: compute the spherical interpolation of two Quat inputs and return the result
	inline Quat slerp(const Quat& q0, const Quat& q1, const double& alpha) {
		return Quat();	// Replace this value with your own code
	}

	// Linear interpolation of two RigTForms
	// TODO: compute the linear interpolation of two RigTForm inputs and return the result
	// Note: you should use the lerp and slerp functions you implemented above
	inline RigTForm Linear(const RigTForm& rbt0, const RigTForm& rbt1, const double& alpha) {
		return RigTForm();	// Replace this value with your own code
	}

}
#endif