/**
 * PCAP Converter Utility
 * Handles conversion of LINUX_SLL (and other non-Ethernet) pcap files to Ethernet format
 * for compatibility with mmt-security and other tools that require native Ethernet headers
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs');
const path = require('path');
const os = require('os');

const execAsync = promisify(exec);

/**
 * Detect the link-layer type of a pcap file
 * @param {string} pcapPath - Path to the pcap file
 * @returns {Promise<{linkType: string, needsConversion: boolean}>}
 */
async function detectLinkType(pcapPath) {
  try {
    // Use capinfos (from wireshark-common) to detect link type
    const { stdout } = await execAsync(`capinfos -T "${pcapPath}" 2>/dev/null || echo "unknown"`);
    const linkType = stdout.trim();
    
    // Common link types that need conversion to Ethernet:
    // - LINUX_SLL (113): Linux cooked capture
    // - LINUX_SLL2 (276): Linux cooked capture v2
    // - RAW (101): Raw IP
    const needsConversion = linkType.includes('LINUX_SLL') || 
                           linkType.includes('Raw IP') ||
                           linkType === '113' || 
                           linkType === '276' ||
                           linkType === '101';
    
    return {
      linkType,
      needsConversion
    };
  } catch (error) {
    console.warn('[PCAP Converter] Could not detect link type, assuming conversion needed:', error.message);
    // If we can't detect, assume conversion might be needed
    return {
      linkType: 'unknown',
      needsConversion: true
    };
  }
}

/**
 * Convert LINUX_SLL or other non-Ethernet pcap to Ethernet format
 * Uses editcap from wireshark-common package
 * @param {string} inputPath - Path to input pcap file
 * @param {string} outputPath - Path for output converted pcap (optional)
 * @returns {Promise<string>} - Path to converted pcap file
 */
async function convertToEthernet(inputPath, outputPath = null) {
  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input pcap file not found: ${inputPath}`);
  }

  // Generate output path if not provided
  if (!outputPath) {
    const tmpDir = os.tmpdir();
    const basename = path.basename(inputPath, path.extname(inputPath));
    const timestamp = Date.now();
    outputPath = path.join(tmpDir, `${basename}_eth_${timestamp}.pcap`);
  }

  try {
    // Use editcap to convert link-layer type to Ethernet (DLT 1)
    // -T 1 sets the encapsulation type to Ethernet
    const cmd = `editcap -T 1 "${inputPath}" "${outputPath}"`;
    console.log('[PCAP Converter] Converting pcap to Ethernet format:', cmd);
    
    const { stdout, stderr } = await execAsync(cmd);
    
    if (stderr && !stderr.includes('records')) {
      console.warn('[PCAP Converter] Conversion warning:', stderr);
    }
    
    if (!fs.existsSync(outputPath)) {
      throw new Error('Conversion failed: output file not created');
    }
    
    console.log('[PCAP Converter] Conversion successful:', outputPath);
    return outputPath;
    
  } catch (error) {
    // Clean up output file if it was created
    if (outputPath && fs.existsSync(outputPath)) {
      try {
        fs.unlinkSync(outputPath);
      } catch (e) {
        // Ignore cleanup errors
      }
    }
    throw new Error(`Failed to convert pcap: ${error.message}`);
  }
}

/**
 * Prepare pcap file for processing with mmt-security
 * Automatically detects if conversion is needed and performs it
 * @param {string} inputPath - Path to input pcap file
 * @param {boolean} forceConversion - Force conversion even if not detected as needed
 * @returns {Promise<{path: string, converted: boolean, cleanup: Function}>}
 */
async function preparePcapForSecurity(inputPath, forceConversion = false) {
  if (!fs.existsSync(inputPath)) {
    throw new Error(`Input pcap file not found: ${inputPath}`);
  }

  let needsConversion = forceConversion;
  let linkTypeInfo = null;

  if (!forceConversion) {
    // Detect if conversion is needed
    linkTypeInfo = await detectLinkType(inputPath);
    needsConversion = linkTypeInfo.needsConversion;
    
    console.log('[PCAP Converter] Link type detection:', {
      file: path.basename(inputPath),
      linkType: linkTypeInfo.linkType,
      needsConversion
    });
  }

  if (!needsConversion) {
    // No conversion needed, return original path
    return {
      path: inputPath,
      converted: false,
      linkType: linkTypeInfo?.linkType || 'Ethernet',
      cleanup: () => {} // No cleanup needed
    };
  }

  // Convert the pcap
  const convertedPath = await convertToEthernet(inputPath);
  
  return {
    path: convertedPath,
    converted: true,
    linkType: linkTypeInfo?.linkType || 'unknown',
    originalPath: inputPath,
    cleanup: () => {
      // Cleanup function to remove temporary converted file
      if (fs.existsSync(convertedPath)) {
        try {
          fs.unlinkSync(convertedPath);
          console.log('[PCAP Converter] Cleaned up temporary file:', convertedPath);
        } catch (error) {
          console.warn('[PCAP Converter] Failed to cleanup temporary file:', error.message);
        }
      }
    }
  };
}

/**
 * Check if required tools are available
 * @returns {Promise<{available: boolean, missing: string[]}>}
 */
async function checkToolsAvailable() {
  const tools = ['editcap', 'capinfos'];
  const missing = [];
  
  for (const tool of tools) {
    try {
      await execAsync(`which ${tool}`);
    } catch (error) {
      missing.push(tool);
    }
  }
  
  return {
    available: missing.length === 0,
    missing
  };
}

module.exports = {
  detectLinkType,
  convertToEthernet,
  preparePcapForSecurity,
  checkToolsAvailable
};
