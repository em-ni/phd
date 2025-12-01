#version 120

varying vec3 v_worldPos;
varying vec3 v_worldNormal;

uniform sampler2D p3d_Texture0; // The mucosa texture
uniform float texScale;

// Lighting uniforms (we can pass these from Python or hardcode for now)
// Simple directional/point light simulation
uniform vec3 lightPos; // World space light position (camera pos)
uniform vec3 lightColor;
uniform vec3 ambientColor;

// Material properties
uniform float shininess;
uniform vec3 k_specular;
uniform vec3 k_diffuse;

void main() {
    // Triplanar Mapping
    vec3 n = normalize(v_worldNormal);
    vec3 blend_weights = abs(n);
    // Tighten the blending to avoid blurry transitions
    blend_weights = (blend_weights - 0.2) * 7.0;
    blend_weights = max(blend_weights, 0.0);
    // Force sum to 1.0
    blend_weights /= (blend_weights.x + blend_weights.y + blend_weights.z);

    // Texture coordinates for each plane
    vec2 coord1 = v_worldPos.yz * texScale;
    vec2 coord2 = v_worldPos.zx * texScale;
    vec2 coord3 = v_worldPos.xy * texScale;

    // Sample texture
    vec3 col1 = texture2D(p3d_Texture0, coord1).rgb;
    vec3 col2 = texture2D(p3d_Texture0, coord2).rgb;
    vec3 col3 = texture2D(p3d_Texture0, coord3).rgb;

    // Blend
    vec3 textureColor = col1 * blend_weights.x + col2 * blend_weights.y + col3 * blend_weights.z;

    // Lighting (Blinn-Phong)
    vec3 viewDir = normalize(lightPos - v_worldPos); // Assuming light is at camera (0,0,0 relative? No, lightPos is world)
    // Actually, if light is attached to camera, its world pos IS camera pos.
    vec3 lightDir = normalize(lightPos - v_worldPos);
    
    // Ambient
    vec3 ambient = ambientColor * textureColor;
    
    // Diffuse
    float diff = max(dot(n, lightDir), 0.0);
    // Tint diffuse with material color (k_diffuse) but mostly texture
    vec3 diffuse = diff * lightColor * k_diffuse * textureColor;
    
    // Specular
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(n, halfDir), 0.0), shininess);
    vec3 specular = spec * lightColor * k_specular;
    
    // Attenuation (Point Light)
    float distance = length(lightPos - v_worldPos);
    // float attenuation = 1.0 / (1.0 + 0.01 * distance * distance); // Matches quadratic=0.01
    // Let's use the same attenuation as before: 1 / (1 + 0*d + 0.01*d^2)
    float attenuation = 1.0 / (1.0 + 0.01 * distance * distance);
    
    vec3 finalColor = ambient + attenuation * (diffuse + specular);
    
    gl_FragColor = vec4(finalColor, 1.0);
}
