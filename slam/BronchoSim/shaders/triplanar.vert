#version 120

// Uniforms from Panda3D
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix; // Inverse transpose of upper 3x3 of ModelViewMatrix? No, usually ModelMatrix for world space lighting if we do it manually.
// Actually, for world space triplanar, we need World Position and World Normal.
// Panda3D's p3d_Vertex is usually model space.

attribute vec4 p3d_Vertex;
attribute vec3 p3d_Normal;

varying vec3 v_worldPos;
varying vec3 v_worldNormal;

void main() {
    // Calculate world position
    v_worldPos = (p3d_ModelMatrix * p3d_Vertex).xyz;
    
    // Calculate world normal (assuming uniform scaling, otherwise use inverse transpose)
    // p3d_ModelMatrix is mat4, cast to mat3 to rotate normal
    v_worldNormal = normalize(mat3(p3d_ModelMatrix) * p3d_Normal);
    
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
