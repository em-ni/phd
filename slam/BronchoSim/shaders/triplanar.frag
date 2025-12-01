#version 120

varying vec3 v_worldPos;
varying vec3 v_worldNormal;

uniform sampler2D p3d_Texture0; // Diffuse
uniform sampler2D p3d_Texture1; // Normal Map
uniform float texScale;

// Lighting uniforms
uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 ambientColor;

// Material properties
uniform float shininess;
uniform vec3 k_specular;
uniform vec3 k_diffuse;

// Helper to unpack normal from texture [0,1] -> [-1,1]
vec3 unpackNormal(vec3 rgb) {
    return normalize(rgb * 2.0 - 1.0);
}

void main() {
    // Triplanar Mapping Weights
    vec3 n_geom = normalize(v_worldNormal);
    vec3 blend_weights = abs(n_geom);
    blend_weights = (blend_weights - 0.2) * 7.0;
    blend_weights = max(blend_weights, 0.0);
    blend_weights /= (blend_weights.x + blend_weights.y + blend_weights.z);

    // Texture coordinates
    vec2 coord1 = v_worldPos.yz * texScale;
    vec2 coord2 = v_worldPos.zx * texScale;
    vec2 coord3 = v_worldPos.xy * texScale;

    // Sample Diffuse
    vec3 col1 = texture2D(p3d_Texture0, coord1).rgb;
    vec3 col2 = texture2D(p3d_Texture0, coord2).rgb;
    vec3 col3 = texture2D(p3d_Texture0, coord3).rgb;
    vec3 textureColor = col1 * blend_weights.x + col2 * blend_weights.y + col3 * blend_weights.z;

    // Sample Normal Map
    vec3 n1 = unpackNormal(texture2D(p3d_Texture1, coord1).rgb);
    vec3 n2 = unpackNormal(texture2D(p3d_Texture1, coord2).rgb);
    vec3 n3 = unpackNormal(texture2D(p3d_Texture1, coord3).rgb);
    
    // Transform normals to world space (Simplified Whiteout Blending / UDN)
    // Correct way requires Tangent-Bitangent-Normal (TBN) matrix for each plane.
    // Approximation: Assume tangent space aligns with world axes for triplanar.
    // Plane X (YZ): Normal is (1,0,0). Tangent (0,1,0), Bitangent (0,0,1).
    // n1 corresponds to YZ plane.
    vec3 worldNormal1 = vec3(0, n1.y, n1.x); // Swizzle to match plane? 
    // Actually, let's use a simpler perturbation method:
    // Blend the tangent-space normals directly? No.
    
    // Better Approximation for Triplanar Normals:
    // We just want to add "bumpiness".
    // Let's blend the sampled normals and add them to the geometric normal.
    vec3 n_bump = n1 * blend_weights.x + n2 * blend_weights.y + n3 * blend_weights.z;
    
    // Perturb the geometric normal
    // Reduced strength from 0.5 to 0.1 to avoid "rough wall" look
    vec3 n_final = normalize(n_geom + n_bump * 0.1);

    // Lighting (Blinn-Phong) using n_final
    vec3 viewDir = normalize(lightPos - v_worldPos);
    vec3 lightDir = normalize(lightPos - v_worldPos);
    
    // Ambient
    vec3 ambient = ambientColor * textureColor;
    
    // Diffuse
    float diff = max(dot(n_final, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * k_diffuse * textureColor;
    
    // Specular
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(n_final, halfDir), 0.0), shininess);
    vec3 specular = spec * lightColor * k_specular;
    
    // Attenuation
    float distance = length(lightPos - v_worldPos);
    float attenuation = 1.0 / (1.0 + 0.01 * distance * distance);
    
    vec3 finalColor = ambient + attenuation * (diffuse + specular);
    
    gl_FragColor = vec4(finalColor, 1.0);
}
